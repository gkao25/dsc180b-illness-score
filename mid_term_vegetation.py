import numpy as np
import pandas as pd
import networkx as nx

import geopandas as gpd
from shapely.geometry import Point

import folium
from folium.plugins import HeatMap

import seaborn as sns
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



vri_df = pd.read_csv('src_vri_snapshot_2024_03_20.csv')
span_df = pd.read_csv('dev_wings_agg_span_2024_01_01.csv')
gis_2024_1004 = pd.read_csv('gis_weatherstation_shape_2024_10_04.csv')
station_summary_2023_08_02 = pd.read_csv('src_wings_meteorology_station_summary_snapshot_2023_08_02.csv')
windspeed_2023_08_02 = pd.read_csv('src_wings_meteorology_windspeed_snapshot_2023_08_02.csv')


merged_df = pd.merge(station_summary_2023_08_02, gis_2024_1004, right_on= 'weatherstationcode', left_on='station', how='left')

windspeed_grouped_count = windspeed_2023_08_02.groupby(by='station').count()

station_codes = np.array(gis_2024_1004['weatherstationcode'])
merged_station_df = gis_2024_1004.merge(station_summary_2023_08_02, left_on='weatherstationcode', right_on='station', how='left')

merged_df[merged_df['weatherstationcode']=='AMO']['alert'].iloc[0]

prob_lst = []

for station in station_codes:
    station_windspeeds = np.array(windspeed_2023_08_02[windspeed_2023_08_02['station'] == station]['wind_speed'])
    # "alert" might be nan because of less entries in station_ss_df
    has_threshold = True
    try:
        threshold = merged_df[merged_df['weatherstationcode'] == station]['alert'].iloc[0]
    except:
        has_threshold = False
        prob = np.nan
    mean = np.nanmean(station_windspeeds)
    if has_threshold:
        prob = np.mean([1 if x >= threshold else 0 for x in station_windspeeds]) * 100
    count = np.count_nonzero(~np.isnan(station_windspeeds))
    prob_lst.append([station, station_windspeeds, threshold, count, mean, prob])

prob_df = pd.DataFrame(prob_lst)
prob_df.columns = ['station', 'windspeeds', 'threshold', 'count', 'mean', 'probability (%)']

gis_2024_1004['shape'] = gpd.GeoSeries.from_wkt(gis_2024_1004['shape'])
gis_gdf = gpd.GeoDataFrame(gis_2024_1004, geometry='shape').set_crs(epsg=4431).to_crs(epsg=4326)

vri_df['shape'] = gpd.GeoSeries.from_wkt(vri_df['shape'])
vri_gdf = gpd.GeoDataFrame(vri_df, geometry='shape').set_crs(epsg=4326)

span_df['shape'] = gpd.GeoSeries.from_wkt(span_df['shape'])
span_gdf = gpd.GeoDataFrame(span_df, geometry='shape').set_crs(epsg=2230).to_crs(epsg=4326)

gis_gdf = gis_gdf.drop(columns=['shape_srid'])
vri_gdf = vri_gdf.drop(columns=['shape_srid'])
span_gdf = span_gdf.drop(columns=['shape_srid'])

gis_vri_merge = gis_gdf.merge(vri_gdf, left_on='weatherstationcode', right_on='anemometercode')
vri_gdf['centroid'] = vri_gdf['shape'].centroid
vri_gis_sjoin = vri_gdf.sjoin(gis_gdf, how='inner')

prob_merge = vri_gis_sjoin.merge(prob_df, left_on='weatherstationcode', right_on='station').merge(station_summary_2023_08_02, left_on='weatherstationcode', right_on='station')

vri_mapping = {'H': 2, 'M': 1, 'L': 0}
prob_merge['vri_numeric'] = prob_merge['vri'].map(vri_mapping)

span_df_cleaned = span_df.dropna(subset=['station'])
span_vri_prob_merge_df = prob_merge.merge(span_df_cleaned, left_on='anemometercode', right_on='station')

columns_to_keep = [
    'probability (%)',
    'vri_numeric', 'elevation', 'longitude', 'latitude',
    'cust_total', 'cust_lifesupport', 'cust_urgent', 'cust_medicalcert', 'cust_essential',
    'cust_sensitive', 'cust_residential', 'cust_commercial', 'cust_industrial',
    'num_strike_trees', 'buffered_tree_counts', 'exclusive_tree_counts'
]

df_filtered = span_vri_prob_merge_df[columns_to_keep]

columns_to_impute = ['num_strike_trees', 'buffered_tree_counts', 'exclusive_tree_counts']
for col in columns_to_impute:
    min_val, max_val = df_filtered[col].min(), df_filtered[col].max()
    df_filtered[col] = df_filtered[col].apply(
        lambda x: np.random.randint(min_val, max_val + 1) if pd.isna(x) else x
    )

plt.figure(figsize=(10, 5))
sns.histplot(df_filtered['probability (%)'], bins=30, kde=True, color='blue')
plt.title("Distribution of Wildfire Probability (%)")
plt.xlabel("Probability (%)")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(12, 8))
correlation_matrix = df_filtered.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

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

columns_to_impute = ['num_strike_trees', 'buffered_tree_counts', 'exclusive_tree_counts']
for col in columns_to_impute:
    min_val, max_val = df_filtered[col].min(), df_filtered[col].max()
    df_filtered[col] = df_filtered[col].apply(
        lambda x: np.random.randint(min_val, max_val + 1) if pd.isna(x) else x
    )

X = df_filtered.drop(columns=['probability (%)'])
y = df_filtered['probability (%)']

imputer = SimpleImputer(strategy="mean")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    # "Support Vector Regressor": SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
    # "MLP Regressor": MLPRegressor(hidden_layer_sizes=(50, 50), learning_rate_init=0.01, max_iter=300, random_state=42)
}

results = []
for name, model in models.items():
    if name in ["Random Forest", "Linear Regression"]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({"Model": name, "MAE": mae, "MSE": mse, "R² Score": r2})

results_df = pd.DataFrame(results)
results_df

