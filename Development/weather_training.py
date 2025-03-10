import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from ens_preprocessing import clean_data


data_dict = clean_data(input_path="2020-08")
ds = list(data_dict.values())[0]

X_ds = ds.isel(time=slice(0, 15))
y_ds = ds.isel(time=slice(15, None))

if "fire_risk" in ds:
    X_da = X_ds["fire_risk"]
    y_da = y_ds["fire_risk"]
else:
    raise ValueError("The variable 'fire_risk' is not found")

X_np = X_da.values
y_np = y_da.values

if X_np.ndim > 2:
    X_np = X_np.reshape(X_np.shape[0], -1)
if y_np.ndim > 2:
    y_np = y_np.reshape(y_np.shape[0], -1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_np)

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_np, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert to PyTorch tensors and send to device.
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)


train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class AttentionLayer(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(feature_dim, 1)
    
    def forward(self, x):
        attention_scores = self.attention_weights(x)  # Shape: (batch_size, 1)
        attention_scores = torch.softmax(attention_scores, dim=1)

        weighted_features = x * attention_scores  # Shape: (batch_size, feature_dim)
        return weighted_features, attention_scores

class WildfireRiskModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5):
        super(WildfireRiskModel, self).__init__()
        self.attention = AttentionLayer(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        # Apply attention
        weighted_features, attention_scores = self.attention(x)
        out = self.fc1(weighted_features)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out, attention_scores

input_size = X_train.shape[1]
hidden_size = 128 
output_size = 1  
model = WildfireRiskModel(input_size, hidden_size, output_size).to(device)

# Define loss function and optimizer
criterion = nn.MSALoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2 regularization

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs, _ = model(X_val)
        val_loss = criterion(val_outputs, y_val)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss.item():.4f}")

# Model evaluation
model.eval()
with torch.no_grad():
    test_outputs, attention_scores = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

    predictions = test_outputs.cpu().numpy()
    actuals = y_test.cpu().numpy()
    attention_scores = attention_scores.cpu().numpy()

# Model save
torch.save(model.state_dict(), "wildfire_risk_model_with_attention.pth")