# ILLNESS, An Alternative Way To Assess Wildfire Risk

UCSD Data Science Capstone Project
DSC 180AB, Section B14-2
Wildfire Mitigation with SDG&E

Group Members: Gloria Kao, Shentong Li, Neil Sharma
Mentors: Mentors: Kasra Mohammadi, Phi Nguyen


## 📂 Project Structure
- `...`
- `...`
- `mid_term_vegetation.py` → Main script for data (vegetation, geographical, living) processing, visualization, and ML model training.
- `requirements.txt` → List of required Python dependencies.
- `README.md` → Instructions on setup and usage.

**Note:** The datasets are not publicly available, so they cannot be uploaded here. However, you can find their metadata below.

---

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
