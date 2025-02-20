# ILLNESS, An Alternative Way To Assess Wildfire Risk

UCSD Data Science Capstone Project\
DSC 180AB, Section B14-2\
Wildfire Mitigation with SDG&E

**Group Members:** Gloria Kao, Shentong Li, Neil Sharma\
**Mentors:** Mentors: Kasra Mohammadi, Phi Nguyen


## 📂 Project Structure
- `README.md` → Instructions on setup and usage.
- `requirements.txt` → List of required Python dependencies.
- `ens_preprocessing.py` → Main script for downloading and preprocessing the data from SDGE/SDSC. 
- `weather_training.py` → Main script for *weather variables* (wind speed, air humidity, etc.) ML model training. 
- `midterm_energy.py` → Main script for *energy conductor* (type, structure, etc.) data processing, visualization, and ML model training.
- `mid_term_vegetation.py` → Main script for *vegetation, geographical, and living* data processing, visualization, and ML model training.
- Jupyter Notebooks → For testing/development. Due to data security, no cummulative outputs are shown. 

**Note:** The datasets are not publicly available, so they cannot be uploaded here. However, you can find their metadata below.

---

## 🔧 Installation & Environment Setup for Vegetation file (mid_term_vegetation.py) and Energy file (midterm_energy.py)

To ensure a **consistent environment**, we recommend using **Conda** to manage dependencies.

#### **1️⃣ Install Conda**
If you haven't installed Conda, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

#### **2️⃣ Create a Conda Virtual Environment**
Open a terminal and run: 
```
bash
conda create --name wildfire_analysis python=3.9 -y
conda activate wildfire_analysis
```

#### **3️⃣ Install Required Packages**
```
pip install numpy pandas networkx geopandas shapely folium seaborn matplotlib scikit-learn
```

or

```
pip install -r requirements.txt
```

#### **4️⃣ Run the Script**
```
python <filename.py>
```


## 📊 Datasets used
#### **1️⃣ GIS Weather Station Data (`gis_weatherstation_shape_2024_10_04.csv`)**
- **Description:** Contains geographic information about weather stations, including their location, elevation, and administrative details.
- **Rows:** 223  
- **Columns:** 27  
- **Columns Description:**
  - `objectid` → Unique identifier for each record.
  - `weatherstationcode` → Unique code assigned to each weather station.
  - `weatherstationname` → Name of the weather station.
  - `scadartuid` → SCADA (Supervisory Control and Data Acquisition) ID.
  - `structureid` → ID of the physical structure where the station is installed.
  - `nwszone` → NOAA Weather Service zone classification.
  - `district` → The district where the weather station is located.
  - `thomasbrospagegrid` → Grid reference in Thomas Bros. maps.
  - `constructionstatus` → Indicates the operational status (e.g., Active `A`).
  - `creationuser`, `lastuser` → Users who created and last modified the record.
  - `datecreated`, `datemodified` → Timestamps of creation and modification.
  - `structureguid` → GUID for the physical structure.
  - `symbolrotation` → Rotation angle of the station marker.
  - `latitude`, `longitude`, `elevation` → Geographic coordinates and elevation in meters.
  - `twinguid` → Associated twin GUID (if applicable).
  - `hftd`, `hftdidc`, `zone1idc` → High fire threat district classifications.
  - `gdb_geomattr_data` → Additional GIS-related attributes.
  - `globalid` → Unique global identifier for the record.
  - `shape` → **GIS Shape Data** stored in **WKT (Well-Known Text)** format.
  - `shape_srid` → Spatial Reference System Identifier (**SRID 4431**).
  - `snapshot_date` → Date when the data was recorded.

---

#### **2️⃣ Vegetation Risk Index (VRI) Data (`src_vri_snapshot_2024_03_20.csv`)**
- **Description:** Contains VRI risk levels for different locations based on historical wind gusts.
- **Columns:**
  - `anemometercode` → Weather station associated with the VRI data.
  - `gust_99pct`, `gust_95pct`, `gust_max` → Wind gust speeds at different percentiles.
  - `vri_risk` → VRI risk level (Low, Medium, High).
  - `county`, `district` → Geographical region details.
  - `shape` → Polygon representation of VRI areas in GIS.
  - `snapshot_date` → Date when the data was recorded.

---

#### **3️⃣ Meteorology Station Summary (`src_wings_meteorology_station_summary_snapshot_2023_08_02.csv`)**
- **Description:** Provides wind speed alerts and risk levels for different weather stations.
- **Columns:**
  - `station` → Weather station identifier.
  - `vri` → VRI classification (H, M, L).
  - `alert` → Wind speed alert threshold.
  - `max_gust`, `99th`, `95th` → Maximum and percentile-based wind gust speeds.
  - `snapshot_date` → Date when the data was recorded.

---

#### **4️⃣ Wind Speed Data (`src_wings_meteorology_windspeed_snapshot_2023_08_02.csv`)**
- **Description:** Historical wind speed data collected from weather stations.
- **Columns:**
  - `date` → Date of wind speed measurement.
  - `wind_speed` → Recorded wind speed.
  - `station` → Identifier for the weather station.
  - `snapshot_date` → Date when the data was recorded.

---

#### **4️⃣ ens_gfs Weather Data (from [this database](https://sdge.sdsc.edu/data/sdge/))**
- **Description:** Historical weather data collected from weather stations.
- **Columns:**
  - `date` → Date of wind speed measurement.
  - `wind_speed` → Recorded wind speed.
  - `station` → Identifier for the weather station.
  - `snapshot_date` → Date when the data was recorded.
