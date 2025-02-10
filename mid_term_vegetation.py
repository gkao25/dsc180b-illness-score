import numpy as np
import pandas as pd
import networkx as nx

import geopandas as gpd
from shapely.geometry import Point

import folium
from folium.plugins import HeatMap

import seaborn as sns
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore")

# Load datasets
vri_df = pd.read_csv('src_vri_snapshot_2024_03_20.csv')
span_df = pd.read_csv('dev_wings_agg_span_2024_01_01.csv')
gis_df = pd.read_csv('gis_weatherstation_shape_2024_10_04.csv')
station_summary_df = pd.read_csv('src_wings_meteorology_station_summary_snapshot_2023_08_02.csv')
windspeed_df = pd.read_csv('src_wings_meteorology_windspeed_snapshot_2023_08_02.csv')

# Merge GIS and station summary datasets
merged_df = station_summary_df.merge(gis_df, left_on='station', right_on='weatherstationcode', how='left')

# Compute windspeed count per station
windspeed_grouped_count = windspeed_df.groupby(by='station').count()

# Extract station codes
station_codes = gis_df['weatherstationcode'].unique()

# Merge GIS and station summary
merged_station_df = gis_df.merge(station_summary_df, left_on='weatherstationcode', right_on='station', how='left')

# Calculate wildfire probability per station
prob_lst = []

for station in station_codes:
    station_windspeeds = windspeed_df[windspeed_df['station'] == station]['wind_speed'].dropna().to_numpy()

    # Handle missing thresholds
    threshold = merged_df.loc[merged_df['weatherstationcode'] == station, 'alert'].dropna()
    threshold = threshold.iloc[0] if not threshold.empty else np.nan

    if np.isnan(threshold) or station_windspeeds.size == 0:
        prob = np.nan
    else:
        prob = np.mean(station_windspeeds >= threshold) * 100

    count = station_windspeeds.size
    mean = np.nanmean(station_windspeeds) if count > 0 else np.nan

    prob_lst.append([station, station_windspeeds, threshold, count, mean, prob])

# Create probability DataFrame
prob_df = pd.DataFrame(prob_lst, columns=['station', 'windspeeds', 'threshold', 'count', 'mean', 'probability (%)'])

# Convert shapefiles to GeoDataFrames
gis_df['shape'] = gpd.GeoSeries.from_wkt(gis_df['shape'])
gis_gdf = gpd.GeoDataFrame(gis_df, geometry='shape', crs='EPSG:4431').to_crs(epsg=4326)

vri_df['shape'] = gpd.GeoSeries.from_wkt(vri_df['shape'])
vri_gdf = gpd.GeoDataFrame(vri_df, geometry='shape', crs='EPSG:4326')

span_df['shape'] = gpd.GeoSeries.from_wkt(span_df['shape'])
span_gdf = gpd.GeoDataFrame(span_df, geometry='shape', crs='EPSG:2230').to_crs(epsg=4326)

# Remove unnecessary columns
gis_gdf.drop(columns=['shape_srid'], inplace=True, errors='ignore')
vri_gdf.drop(columns=['shape_srid'], inplace=True, errors='ignore')
span_gdf.drop(columns=['shape_srid'], inplace=True, errors='ignore')

# Spatial join between VRI and GIS
vri_gdf['centroid'] = vri_gdf['shape'].centroid
vri_gis_sjoin = vri_gdf.sjoin(gis_gdf, how='inner')

# Merge probability data with wildfire risk
prob_merge = vri_gis_sjoin.merge(prob_df, left_on='weatherstationcode', right_on='station').merge(station_summary_df, left_on='weatherstationcode', right_on='station')

# Map VRI categories to numerical values
vri_mapping = {'H': 2, 'M': 1, 'L': 0}
prob_merge['vri_numeric'] = prob_merge['vri'].map(vri_mapping)

# Clean span dataset and merge
span_df_cleaned = span_df.dropna(subset=['station'])
span_vri_prob_merge_df = prob_merge.merge(span_df_cleaned, left_on='anemometercode', right_on='station')

# Select features
columns_to_keep = [
    'probability (%)', 'vri_numeric', 'elevation', 'longitude', 'latitude',
    'cust_total', 'cust_lifesupport', 'cust_urgent', 'cust_medicalcert',
    'cust_essential', 'cust_sensitive', 'cust_residential', 'cust_commercial',
    'cust_industrial', 'num_strike_trees', 'buffered_tree_counts', 'exclusive_tree_counts'
]

df_filtered = span_vri_prob_merge_df[columns_to_keep]

# Handle missing values with mean imputation
imputer = SimpleImputer(strategy="mean")
df_filtered[columns_to_keep] = imputer.fit_transform(df_filtered)

# Visualization: Wildfire Probability Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df_filtered['probability (%)'], bins=30, kde=True, color='blue')
plt.title("Distribution of Wildfire Probability (%)")
plt.xlabel("Probability (%)")
plt.ylabel("Frequency")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df_filtered.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Scatterplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.scatterplot(data=df_filtered, x='elevation', y='probability (%)', ax=axes[0, 0], color='blue')
axes[0, 0].set_title("Elevation vs Wildfire Probability")

sns.scatterplot(data=df_filtered, x='vri_numeric', y='probability (%)', ax=axes[0, 1], color='red')
axes[0, 1].set_title("VRI Risk vs Wildfire Probability")

sns.scatterplot(data=df_filtered, x='cust_total', y='probability (%)', ax=axes[1, 0], color='green')
axes[1, 0].set_title("Total Customers vs Wildfire Probability")

sns.scatterplot(data=df_filtered, x='num_strike_trees', y='probability (%)', ax=axes[1, 1], color='orange')
axes[1, 1].set_title("Number of Trees vs Wildfire Probability")

plt.tight_layout()
plt.show()

# Machine Learning Model Training
X = df_filtered.drop(columns=['probability (%)'])
y = df_filtered['probability (%)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({"Model": name, "MAE": mean_absolute_error(y_test, y_pred), "MSE": mean_squared_error(y_test, y_pred), "R² Score": r2_score(y_test, y_pred)})

results_df = pd.DataFrame(results)
print(results_df)
