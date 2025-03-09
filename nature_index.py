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
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.impute import SimpleImputer


import warnings
warnings.filterwarnings("ignore")
vri_df = pd.read_csv('Desktop/UCSD/DSC_180B/living_life_code/src_vri_snapshot_2024_03_20.csv')
span_df = pd.read_csv('Desktop/UCSD/DSC_180B/living_life_code/dev_wings_agg_span_2024_01_01.csv')
gis_2024_1004 = pd.read_csv('Desktop/UCSD/DSC_180B/living_life_code/gis_weatherstation_shape_2024_10_04.csv')
station_summary_2023_08_02 = pd.read_csv('Desktop/UCSD/DSC_180B/living_life_code/src_wings_meteorology_station_summary_snapshot_2023_08_02.csv')
windspeed_2023_08_02 = pd.read_csv('Desktop/UCSD/DSC_180B/living_life_code/src_wings_meteorology_windspeed_snapshot_2023_08_02.csv')


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


columns_to_keep = ['probability (%)', 'vri_numeric', 'elevation', 'longitude', 'latitude', 'num_strike_trees', 'shape_y']
df_filtered = span_vri_prob_merge_df[columns_to_keep]

columns_to_keep = ['probability (%)', 'vri_numeric', 'elevation', 'longitude', 'latitude', 'num_strike_trees', 'shape_y']
df_filtered = span_vri_prob_merge_df[columns_to_keep].dropna()  # Remove NaN values

# Drop duplicates based on 'latitude' and 'longitude'
df_filtered = df_filtered.drop_duplicates(subset=['latitude', 'longitude'])
# df_filtered1 = df_filtered[['probability (%)', 'vri_numeric', 'elevation', 'longitude', 'latitude', 'hftd', 'num_strike_trees']]
df_filtered1 = df_filtered[['probability (%)', 'vri_numeric', 'elevation', 'longitude', 'latitude', 'num_strike_trees']]
correlation_matrix = df_filtered1.corr()

features = ['vri_numeric', 'elevation', 'longitude', 'latitude', 'num_strike_trees']
target = 'probability (%)'

X = df_filtered[features]
y = df_filtered[target]

model = LinearRegression()
model.fit(X, y)

# Extract absolute feature weights and normalize them to sum to 1
feature_weights = np.abs(model.coef_)
feature_weights /= np.sum(feature_weights)  # Normalize weights

# Compute the Nature Index as a weighted sum of the features
df_filtered['nature_index'] = np.dot(X, feature_weights)

# Scale the Nature Index to a range of 1-10
scaler = MinMaxScaler(feature_range=(1, 10))
df_filtered['nature_index'] = scaler.fit_transform(df_filtered[['nature_index']])

# Create a DataFrame for feature weights
feature_weights_df = pd.DataFrame({'Feature': features, 'Weight': feature_weights})
feature_weights_df.sort_values(by='Weight', ascending=False, inplace=True)

df_filtered_export = df_filtered[['nature_index', 'longitude', 'latitude', 'shape_y']]
