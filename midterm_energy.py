import numpy as np
import pandas as pd
import networkx as nx

import geopandas as gpd
from shapely.geometry import Point

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
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings("ignore")


### Preprocess the data

## Load the datasets
span_df = pd.read_csv('data/dev_wings_agg_span_2024_01_01.csv')
gis_2024_1004 = pd.read_csv('data/gis_weatherstation_shape_2024_10_04.csv')
station_summary_2023_08_02 = pd.read_csv('data/src_wings_meteorology_station_summary_snapshot_2023_08_02.csv')
windspeed_2023_08_02 = pd.read_csv('data/src_wings_meteorology_windspeed_snapshot_2023_08_02.csv')

## Merge the datasets
merged_df = pd.merge(station_summary_2023_08_02, gis_2024_1004, right_on= 'weatherstationcode', left_on='station', how='left')
windspeed_grouped_count = windspeed_2023_08_02.groupby(by='station').count()
station_codes = np.array(gis_2024_1004['weatherstationcode'])
merged_station_df = gis_2024_1004.merge(station_summary_2023_08_02, left_on='weatherstationcode', right_on='station', how='left')

## Calculate the PSPS possibilities
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

# Create the PSPS prob dataframe
prob_df = pd.DataFrame(prob_lst)
prob_df.columns = ['station', 'windspeeds', 'threshold', 'count', 'mean', 'probability (%)']

## Convert the dataframe types to work with geographical data, making sure they use the same coordinate system
gis_2024_1004['shape'] = gpd.GeoSeries.from_wkt(gis_2024_1004['shape'])
gis_gdf = gpd.GeoDataFrame(gis_2024_1004, geometry='shape').set_crs(epsg=4431).to_crs(epsg=4326)

span_df['shape'] = gpd.GeoSeries.from_wkt(span_df['shape'])
span_gdf = gpd.GeoDataFrame(span_df, geometry='shape').set_crs(epsg=2230).to_crs(epsg=4326)

gis_gdf = gis_gdf.drop(columns=['shape_srid'])
span_gdf = span_gdf.drop(columns=['shape_srid'])

## Merge dataframes with PSPS prob based on weather station 
prob_merge = gis_gdf.merge(prob_df, left_on='weatherstationcode', right_on='station').merge(station_summary_2023_08_02, left_on='weatherstationcode', right_on='station')

## Merge with conductor span dataframe
span_df_cleaned = span_df.dropna(subset=['station'])
span_vri_prob_merge_df = prob_merge.merge(span_df_cleaned, left_on='weatherstationcode', right_on='station')

## Keep only the conductor related columns that we are interested in 
columns_to_keep = [
    'probability (%)',
    'hardened_state', 'miles', 
    'upstream_struct_age', 'upstream_struct_hftd', 'upstream_struct_material', 'upstream_struct_type', 'upstream_struct_workorderdate',
    'downstream_struct_age', 'downstream_struct_hftd', 'downstream_struct_material', 'downstream_struct_type', 'downstream_struct_workorderdate',
    'wire_risk',
]
df_filtered = span_vri_prob_merge_df[columns_to_keep]

## Fix 'workorderdate' datatype from str to datetime
df_filtered['upstream_struct_workorderdate'] = df_filtered['upstream_struct_workorderdate'].replace('NaN',np.nan)
df_filtered['downstream_struct_workorderdate'] = df_filtered['downstream_struct_workorderdate'].replace('NaN',np.nan)

df_filtered['upstream_struct_workorderdate'] = pd.to_datetime(df_filtered['upstream_struct_workorderdate'],format='%Y-%m-%d')
df_filtered['downstream_struct_workorderdate'] = pd.to_datetime(df_filtered['downstream_struct_workorderdate'],format='%Y-%m-%d')

# Calculate the days past since the last work order
upstream_workorder_days = pd.Timestamp.now() - df_filtered['upstream_struct_workorderdate']
df_filtered['days_since_upstream_workorder'] = [i.days for i in upstream_workorder_days]
downstream_workorder_days = pd.Timestamp.now() - df_filtered['downstream_struct_workorderdate']
df_filtered['days_since_downstream_workorder'] = [i.days for i in downstream_workorder_days]

# Drop original columns
df_filtered = df_filtered.drop(columns=['upstream_struct_workorderdate', 'downstream_struct_workorderdate'])

## One-hot encoding
encoder = OneHotEncoder(sparse_output=False)
categorical_columns = ['hardened_state', 'wire_risk',
                       'upstream_struct_type', 'downstream_struct_type', 'downstream_struct_material', 'upstream_struct_material']
one_hot_encoded = encoder.fit_transform(df_filtered[categorical_columns])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

df_encoded = pd.concat([df_filtered.drop(categorical_columns, axis=1), one_hot_df], axis=1)


### EDA data visualizations
# Bar plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sorted_type = df_filtered['upstream_struct_type'].unique()
sorted_material = df_filtered['upstream_struct_material'].unique()

sns.barplot(data=df_filtered, x='upstream_struct_type', y='probability (%)', order=sorted_type, ax=axes[0, 0], color='blue')
axes[0, 0].set_title("Upstream Structure Type vs Wildfire Probability")

sns.barplot(data=df_filtered, x='downstream_struct_type', y='probability (%)', order=sorted_type, ax=axes[1, 0], color='red')
axes[1, 0].set_title("Downstream Struct Type vs Wildfire Probability")

sns.barplot(data=df_filtered, x='upstream_struct_material', y='probability (%)', order=sorted_material, ax=axes[0, 1], color='blue')
axes[0, 1].set_title("Upstream Structure Material vs Wildfire Probability")

sns.barplot(data=df_filtered, x='downstream_struct_material', y='probability (%)', order=sorted_material, ax=axes[1, 1], color='red')
axes[1, 1].set_title("Downstream Struct Material vs Wildfire Probability")

plt.tight_layout()
plt.savefig('spans_structure_barplots.png', dpi=300)

# Scatter plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.scatterplot(data=df_filtered, x='wire_risk', y='probability (%)', ax=axes[0, 0], color='orange')
axes[0, 0].set_title("Wire Risk vs Wildfire Probability")

sns.scatterplot(data=df_filtered, x='upstream_struct_age', y='probability (%)', ax=axes[0, 1], color='blue')
axes[0, 1].set_title("Upstream Struct Age vs Wildfire Probability")

sns.scatterplot(data=df_filtered, x='miles', y='probability (%)', ax=axes[1, 0], color='green')
axes[1, 0].set_title("Miles vs Wildfire Probability")

sns.scatterplot(data=df_filtered, x='downstream_struct_age', y='probability (%)', ax=axes[1, 1], color='red')
axes[1, 1].set_title("Downstream Struct Age vs Wildfire Probability")

plt.tight_layout()
plt.savefig('spans_scatterplots.png', dpi=300)


## ML models
X = df_encoded.drop(columns=['probability (%)'])
y = df_encoded['probability (%)']

# Impute the missing values
imputer = SimpleImputer(strategy="mean")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=25, random_state=42),
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
print(results_df)