# ILLNESS, An Alternative Way To Assess Wildfire Risk

UCSD Data Science Capstone Project
DSC 180AB, Section B14-2
Wildfire Mitigation with SDG&E

Group Members: Gloria Kao, Shentong Li, Neil Sharma
Mentors: Mentors: Kasra Mohammadi, Phi Nguyen


## 📂 Project Structure
- `...`
- `...`
- `mid_term_vegetation.py` → Main script for (vegetation, geographical, living) data processing, visualization, and ML model training.
- `requirements.txt` → List of required Python dependencies.
- `README.md` → Instructions on setup and usage.

**Note:** The datasets are not publicly available, so they cannot be uploaded here. However, you can find their metadata below.

---
## 📊 Datasets used
##### **1️⃣ GIS Weather Station Data (`gis_weatherstation_shape_2024_10_04.csv`)**
- **Description:** Contains geographic information about weather stations, including their location, elevation, and identifiers.
- **Columns:**
  - `weatherstationcode` → Unique identifier for each station.
  - `weatherstationname` → Name of the weather station.
  - `latitude`, `longitude`, `elevation` → Geographic coordinates and elevation of the station.
  - `district`, `nwszone` → Administrative and weather zone classifications.
  - `shape` → GIS shape data in **WKT (Well-Known Text)** format.
  - `snapshot_date` → Date when the data was recorded.

---

##### **2️⃣ Vegetation Risk Index (VRI) Data (`src_vri_snapshot_2024_03_20.csv`)**
- **Description:** Contains VRI risk levels for different locations based on historical wind gusts.
- **Columns:**
  - `anemometercode` → Weather station associated with the VRI data.
  - `gust_99pct`, `gust_95pct`, `gust_max` → Wind gust speeds at different percentiles.
  - `vri_risk` → VRI risk level (Low, Medium, High).
  - `county`, `district` → Geographical region details.
  - `shape` → Polygon representation of VRI areas in GIS.
  - `snapshot_date` → Date when the data was recorded.

---

##### **3️⃣ Meteorology Station Summary (`src_wings_meteorology_station_summary_snapshot_2023_08_02.csv`)**
- **Description:** Provides wind speed alerts and risk levels for different weather stations.
- **Columns:**
  - `station` → Weather station identifier.
  - `vri` → VRI classification (H, M, L).
  - `alert` → Wind speed alert threshold.
  - `max_gust`, `99th`, `95th` → Maximum and percentile-based wind gust speeds.
  - `snapshot_date` → Date when the data was recorded.

---

##### **4️⃣ Wind Speed Data (`src_wings_meteorology_windspeed_snapshot_2023_08_02.csv`)**
- **Description:** Historical wind speed data collected from weather stations.
- **Columns:**
  - `date` → Date of wind speed measurement.
  - `wind_speed` → Recorded wind speed.
  - `station` → Identifier for the weather station.
  - `snapshot_date` → Date when the data was recorded.

## 🔧 Installation & Environment Setup for Vegetation file (mid_term_vegetation.py)

To ensure a **consistent environment**, we recommend using **Conda** to manage dependencies.

### **1️⃣ Install Conda**
If you haven't installed Conda, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

### **2️⃣ Create a Conda Virtual Environment**
Open a terminal and run: ```bash

conda create --name wildfire_analysis python=3.9 -y

conda activate wildfire_analysis

### **3️⃣ Install Required Packages**
pip install numpy pandas networkx geopandas shapely folium seaborn matplotlib scikit-learn 

or

pip install -r requirements.txt

### **4️⃣ Run the Script**
python mid_term_vegetation.py
