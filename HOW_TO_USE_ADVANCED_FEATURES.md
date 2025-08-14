# 🚀 How to Use Advanced Features

## Overview
Your Solar Power Generation Predictor now includes **5 advanced features** that are fully integrated into the Streamlit web interface. Here's how to access and use each one:

## 🌐 Accessing the Application
1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and go to the URL shown (usually `http://localhost:8501`)

3. **Enter a city name** in the sidebar (e.g., "London", "New York", "Tokyo")

## 📋 Available Features (6 Tabs)

### 1. 🔍 Basic Prediction
- **What it does:** Original solar power prediction using a single ML model
- **How to use:** 
  - Enter city name in sidebar
  - Click "🔍 Get Basic Prediction"
  - View power generation gauge and weather conditions

### 2. 🤖 Ensemble ML (Advanced Feature #1)
- **What it does:** Uses multiple ML models and dynamically selects the best one based on weather conditions
- **How to use:**
  - Enter city name in sidebar
  - Click "🤖 Get Ensemble Prediction"
  - View ensemble prediction with confidence scores
  - See predictions from all individual models
- **Prerequisites:** Run `python ensemble_trainer.py` first to train ensemble models

### 3. 💰 ROI Calculator (Advanced Feature #2)
- **What it does:** Calculates return on investment, payback period, NPV, IRR, and environmental benefits
- **How to use:**
  - Set system size (kW) and daily generation (kWh)
  - Choose your region
  - Set analysis period (years)
  - Click "💰 Calculate ROI"
  - View financial metrics, environmental impact, and annual breakdown

### 4. 📊 3D Visualizations (Advanced Feature #3)
- **What it does:** Interactive 3D visualizations of solar potential and irradiance
- **How to use:**
  - Choose visualization type:
    - Daily Irradiance Profile
    - Weather Impact Analysis
    - Solar Path Visualization
    - Seasonal Comparison
  - Click "📊 Generate Visualization"
  - Interact with 3D plots and charts

### 5. 🛰️ Satellite Data (Advanced Feature #4)
- **What it does:** Enhanced weather data combining satellite imagery with ground-based stations
- **How to use:**
  - Click "🛰️ Get Enhanced Weather Data"
  - View enhanced cloud cover, temperature, and satellite details
  - Click "📊 Analyze Cloud Trends" for historical analysis
- **Optional:** Add NASA API key to `.env` for real satellite data

### 6. 🌱 Carbon Tracker (Advanced Feature #5)
- **What it does:** Tracks CO2 emissions saved and environmental impact
- **How to use:**
  - Enter solar generation (kWh) and system size (kW)
  - Choose region and time period
  - Click "🌱 Calculate Carbon Impact"
  - View CO2 savings, environmental equivalencies, and regional comparisons

## 🎯 Feature Highlights

### 🤖 Multi-Model Ensemble
- **5 different ML models:** Random Forest, Gradient Boosting, XGBoost, SVR, Linear Regression
- **Dynamic selection:** Meta-learner chooses the best model for current weather conditions
- **Confidence scoring:** Shows prediction confidence based on model agreement

### 💰 ROI Calculator
- **Financial metrics:** ROI, payback period, NPV, IRR, LCOE
- **Regional pricing:** Different electricity prices by region
- **Environmental benefits:** CO2 savings, trees equivalent, cars off road
- **Annual breakdown:** Detailed cash flow analysis

### 📊 3D Visualizations
- **Interactive plots:** Zoom, rotate, and explore 3D solar data
- **Multiple views:** Daily profiles, seasonal comparisons, weather impact
- **Real-time updates:** Based on current weather conditions

### 🛰️ Satellite Data
- **Enhanced accuracy:** Combines satellite and ground-based weather data
- **Cloud analysis:** Detailed cloud cover and type information
- **Trend analysis:** Historical cloud cover patterns

### 🌱 Carbon Tracker
- **Environmental impact:** CO2 savings and environmental equivalencies
- **Regional comparison:** Compare impact across different grid mixes
- **System efficiency:** Performance analysis and optimization

## 🔧 Setup Requirements

### Required Files
- `.env` file with your OpenWeatherMap API key
- Trained models (run `python simple_trainer.py` for basic model)
- Ensemble models (run `python ensemble_trainer.py` for advanced features)

### Optional Setup
- NASA API key for real satellite data
- Earthdata credentials for MODIS satellite data

## 🎨 User Interface Features

### Visual Elements
- **Gauge charts:** Power generation visualization
- **Radar charts:** Weather conditions overview
- **3D plots:** Interactive solar irradiance maps
- **Bar charts:** Regional comparisons and metrics
- **Color-coded cards:** Different feature types

### Responsive Design
- **Wide layout:** Optimized for desktop viewing
- **Sidebar navigation:** Easy access to all features
- **Tabbed interface:** Organized feature access
- **Real-time updates:** Live weather data integration

## 🚀 Getting Started

1. **First time setup:**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Set up environment
   cp .env.example .env
   # Edit .env and add your API keys
   
   # Train models
   python simple_trainer.py
   python ensemble_trainer.py
   ```

2. **Run the application:**
   ```bash
   streamlit run app.py
   ```

3. **Explore features:**
   - Start with Basic Prediction to understand the core functionality
   - Try Ensemble ML for more accurate predictions
   - Use ROI Calculator for financial analysis
   - Explore 3D Visualizations for solar potential
   - Check Satellite Data for enhanced weather information
   - Use Carbon Tracker for environmental impact

## 🎯 Pro Tips

- **City names:** Use major cities for best results (London, New York, Tokyo, etc.)
- **Weather data:** Data is cached for 5 minutes to avoid API rate limits
- **Visualizations:** Use full-screen mode for better 3D interaction
- **ROI calculations:** Adjust parameters to match your specific situation
- **Satellite data:** Works best with clear weather conditions

## 🔍 Troubleshooting

- **"Model not found" errors:** Run the training scripts first
- **API errors:** Check your `.env` file and API key validity
- **Import errors:** Ensure all dependencies are installed
- **Visualization issues:** Try refreshing the page or using a different browser

---

**Enjoy exploring all the advanced features of your Solar Power Generation Predictor!** ☀️