import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch_optimizer import Lamb, Ranger
import glob
import pandas as pd
import os

class HumidityAttentionMLP(nn.Module):
    def __init__(self, num_other_features=8, embed_dim=128, num_heads=8, mlp_hidden=512, dropout=0.2):
        super().__init__()
        self.humidity_embedding = nn.Linear(1, embed_dim)
        self.other_embedding = nn.Linear(1, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),  # Additional layer
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, 1)

    def forward(self, humidity, other_features):
        humidity = humidity.unsqueeze(-1)
        humidity_emb = self.humidity_embedding(humidity)
        other_features = other_features.unsqueeze(-1)
        other_emb = self.other_embedding(other_features)
        attn_output, attn_weights = self.attention(humidity_emb, other_emb, other_emb)
        attn_output = self.norm1(attn_output.squeeze(1))
        x = self.fc1(attn_output)
        x = self.norm2(x)
        out = self.fc2(x)
        return out, attn_weights
    
class FireRiskDataset(Dataset):
    def __init__(self, humidity_tensor, other_tensor, label_tensor):
        self.humidity = humidity_tensor
        self.other_features = other_tensor
        self.labels = label_tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.humidity[idx], self.other_features[idx], self.labels[idx]
    
def combine(folder_path):
    df_list = []

    for subfolder in [f.path for f in os.scandir(folder_path) if f.is_dir()]:
        csv_files = glob.glob(os.path.join(subfolder, "*.csv"))
        for file in csv_files:
            temp_df = pd.read_csv(file)  # Adjust 'sep' if needed

            # Process lat_lon into latitude and longitude
            temp_df[['latitude', 'longitude']] = (
                temp_df['lat_lon']
                .str.strip('()')  # Remove parentheses
                .str.split(',', expand=True)
                .apply(lambda col: pd.to_numeric(col, errors='coerce'))
            )

            temp_df.drop('lat_lon', axis=1, inplace=True)

            df_list.append(temp_df)
    final_df = pd.concat(df_list, ignore_index=True)

    return final_df

total_data = combine('Cleaned_ENS')

other_features_cols = [
    'mean_wtd_moisture_1hr', 'mean_wtd_moisture_10hr',
    'air_temperature_2m', 'wind_speed',
    'accumulated_precipitation_amount', 'surface_downwelling_shortwave_flux'
]

# Columns for latitude and longitude
geo_features_cols = ['latitude', 'longitude']

from sklearn.model_selection import train_test_split

# Split dataset into train (80%) and validation (20%)
train_data, val_data = train_test_split(total_data, test_size=0.2, random_state=42)

# Normalize input features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_other = torch.tensor(scaler.fit_transform(train_data[other_features_cols].values), dtype=torch.float32)
val_other = torch.tensor(scaler.transform(val_data[other_features_cols].values), dtype=torch.float32)

# Keep latitude and longitude in their original form
train_geo = torch.tensor(train_data[geo_features_cols].values, dtype=torch.float32)
val_geo = torch.tensor(val_data[geo_features_cols].values, dtype=torch.float32)

# Combine scaled features with unscaled latitude and longitude
train_other = torch.cat([train_other, train_geo], dim=1)
val_other = torch.cat([val_other, val_geo], dim=1)

train_humidity = torch.tensor(scaler.fit_transform(train_data[['air_relative_humidity_2m']].values), dtype=torch.float32)
val_humidity = torch.tensor(scaler.transform(val_data[['air_relative_humidity_2m']].values), dtype=torch.float32)

# Normalize target labels
train_labels = torch.tensor(scaler.fit_transform(train_data[['fire_risk_score']].values), dtype=torch.float32)
val_labels = torch.tensor(scaler.transform(val_data[['fire_risk_score']].values), dtype=torch.float32)

train_dataset = FireRiskDataset(train_humidity, train_other, train_labels)
val_dataset = FireRiskDataset(val_humidity, val_other, val_labels)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Verify tensor shapes
print("Humidity tensor shape:", train_humidity.shape)  # (num_samples,)
print("Other features tensor shape:", train_other.shape)  # (num_samples, 8)
print("Labels tensor shape:", train_labels.shape)  # (num_samples, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Initialize the model and move it to GPU
model = HumidityAttentionMLP(num_other_features=8, embed_dim=16, num_heads=2, mlp_hidden=32)
model.to(device)

# Define loss function and optimizer
criterion = nn.HuberLoss()
optimizer = Lamb(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

num_epochs = 10
train_losses = []
val_losses = []

# Early stopping
best_val_loss = float('inf')
patience = 5
epochs_without_improvement = 0

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0

    for batch_humidity, batch_other, batch_labels in train_loader:
        batch_humidity, batch_other, batch_labels = batch_humidity.to(device), batch_other.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(batch_humidity, batch_other)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item() * batch_labels.size(0)

    epoch_train_loss = total_train_loss / len(train_dataset)
    train_losses.append(epoch_train_loss)

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch_humidity, batch_other, batch_labels in val_loader:
            batch_humidity, batch_other, batch_labels = batch_humidity.to(device), batch_other.to(device), batch_labels.to(device)
            outputs, _ = model(batch_humidity, batch_other)
            loss = criterion(outputs, batch_labels)
            total_val_loss += loss.item() * batch_labels.size(0)

    epoch_val_loss = total_val_loss / len(val_dataset)
    val_losses.append(epoch_val_loss)

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping triggered!")
        break

    scheduler.step()