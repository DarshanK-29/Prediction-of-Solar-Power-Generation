# 🌟 Advanced Solar Power Prediction App - User Guide

## 🚀 Quick Start

1. **Setup**: Make sure you have your OpenWeatherMap API key in the `.env` file
2. **Run**: `streamlit run app.py`
3. **Use**: Enter a city name and explore the 6 different tabs!

## 📋 Available Features

### 1. 🔍 Basic Prediction Tab
**What it does**: Traditional solar power prediction using a single ML model
- Enter a city name
- Get real-time weather data
- View predicted solar power generation
- See weather conditions radar chart
- Check model confidence and information

**Best for**: Quick, simple predictions

### 2. 🤖 Ensemble ML Tab
**What it does**: Advanced prediction using multiple ML models with dynamic selection
- Uses 5 different ML models (Random Forest, Gradient Boosting, XGBoost, SVR, Linear Regression)
- Dynamically selects the best model based on weather conditions
- Shows predictions from all models
- Provides higher accuracy and confidence

**Best for**: Most accurate predictions, research purposes

### 3. 💰 ROI Calculator Tab
**What it does**: Calculate financial returns on solar installations
- **Input**: System size, daily generation, region, analysis period
- **Output**: ROI percentage, payback period, total savings, NPV, IRR
- **Environmental**: CO2 saved, trees equivalent, cars off road
- **Regional**: Different electricity prices and incentives by region

**Best for**: Financial planning, investment decisions

**Example Results**:
- 5 kW system with 20 kWh daily generation
- 232.8% ROI over 25 years
- $45,000+ total savings
- 1,200+ kg CO2 saved annually

### 4. 📊 3D Visualizations Tab
**What it does**: Interactive 3D and 2D solar irradiance visualizations
- **Daily Irradiance Profile**: 24-hour solar potential
- **Weather Impact Analysis**: How clouds affect solar generation
- **Solar Path Visualization**: Sun's path throughout the day
- **Seasonal Comparison**: Solar potential across seasons

**Best for**: Understanding solar patterns, educational purposes

### 5. 🛰️ Satellite Data Tab
**What it does**: Enhanced weather data using satellite imagery
- Combines ground weather stations with satellite data
- More accurate cloud cover information
- Cloud trend analysis over time
- Multiple data sources for reliability

**Best for**: More accurate weather data, research

### 6. 🌱 Carbon Tracker Tab
**What it does**: Track environmental impact of solar generation
- Calculate CO2 emissions saved
- Environmental equivalencies (trees, cars, flights)
- Regional grid mix comparisons
- System efficiency analysis

**Best for**: Environmental impact assessment, sustainability reporting

## 🎯 How to Use Each Feature

### Basic Prediction
1. Go to "🔍 Basic Prediction" tab
2. Enter city name in sidebar
3. Click "🔍 Get Basic Prediction"
4. View results and visualizations

### Ensemble ML
1. Go to "🤖 Ensemble ML" tab
2. Enter city name in sidebar
3. Click "🤖 Get Ensemble Prediction"
4. Compare predictions from all models
5. See which model was selected and why

### ROI Calculator
1. Go to "💰 ROI Calculator" tab
2. Set system parameters:
   - System size (kW)
   - Daily generation (kWh)
   - Region (affects electricity prices)
   - Analysis period (years)
3. Click "💰 Calculate ROI"
4. Review financial and environmental metrics

### 3D Visualizations
1. Go to "📊 3D Visualizations" tab
2. Enter city name in sidebar
3. Choose visualization type from dropdown
4. Click "📊 Generate Visualization"
5. Interact with the 3D charts

### Satellite Data
1. Go to "🛰️ Satellite Data" tab
2. Enter city name in sidebar
3. Click "🛰️ Get Enhanced Weather Data"
4. View enhanced weather information
5. Optional: Click "📊 Analyze Cloud Trends"

### Carbon Tracker
1. Go to "🌱 Carbon Tracker" tab
2. Set parameters:
   - Solar generation (kWh)
   - Region
   - Time period
   - System size
3. Click "🌱 Calculate Carbon Impact"
4. View environmental metrics
5. Optional: Compare regions or analyze efficiency

## 🔧 Technical Details

### Supported Cities
The app works with any city that OpenWeatherMap API supports. Popular cities include:
- London, New York, Tokyo, Paris, Sydney
- Mumbai, Beijing, Berlin, Rome, Madrid
- And thousands more!

### API Requirements
- **OpenWeatherMap API**: Required for weather data
- **NASA API**: Optional for enhanced satellite data
- **Earthdata**: Optional for MODIS satellite data

### Model Information
- **Basic Model**: Single Random Forest model
- **Ensemble Models**: 5 different ML algorithms
- **Training Data**: 2 years of synthetic solar data
- **Features**: 24 engineered features including weather, time, and interactions

## 🎨 Visualizations Explained

### Power Gauge
- **Green**: High generation (70-100% of capacity)
- **Yellow**: Medium generation (30-70% of capacity)
- **Gray**: Low generation (0-30% of capacity)

### Weather Radar
- Shows normalized weather conditions
- Helps understand which factors affect generation most

### 3D Maps
- **X-axis**: Time of day
- **Y-axis**: Day of year
- **Z-axis**: Solar irradiance (W/m²)
- **Color**: Irradiance intensity

## 💡 Tips for Best Results

1. **For Accurate Predictions**: Use the Ensemble ML tab
2. **For Financial Planning**: Use ROI Calculator with realistic generation estimates
3. **For Understanding Patterns**: Use 3D Visualizations
4. **For Research**: Combine Satellite Data with Ensemble predictions
5. **For Sustainability**: Use Carbon Tracker to measure environmental impact

## 🚨 Troubleshooting

### "API key not found" Error
- Check that your `.env` file exists
- Ensure `OPENWEATHER_API_KEY=your_key_here` is in the file
- Restart the application

### "Model not found" Error
- Run `python simple_trainer.py` to train basic model
- Run `python ensemble_trainer.py` to train ensemble models

### "Feature not available" Warning
- Some features require additional API keys (NASA, Earthdata)
- Features will work with fallback data if APIs are not configured

### Slow Loading
- Weather data is cached for 5 minutes
- Large visualizations may take a few seconds to generate
- Ensemble predictions use multiple models, so they take longer

## 📊 Understanding the Results

### Power Generation
- **Units**: Kilowatts (kW)
- **Range**: 0-100 kW (configurable)
- **Realistic**: Accounts for time of day, weather, and seasonal effects

### Confidence Scores
- **High (80-100%)**: Very reliable prediction
- **Medium (60-80%)**: Good reliability
- **Low (40-60%)**: Moderate reliability
- **Very Low (<40%)**: Less reliable, consider other factors

### Financial Metrics
- **ROI**: Return on investment percentage
- **Payback Period**: Years to recover installation cost
- **NPV**: Net present value of the investment
- **LCOE**: Levelized cost of energy

### Environmental Impact
- **CO2 Saved**: Kilograms of carbon dioxide avoided
- **Trees Equivalent**: Number of trees needed to absorb same CO2
- **Cars Off Road**: Equivalent cars removed from road

## 🎓 Educational Use Cases

1. **Solar Energy Education**: Use 3D visualizations to teach solar patterns
2. **Financial Literacy**: Use ROI calculator to understand investment analysis
3. **Environmental Science**: Use carbon tracker to understand climate impact
4. **Data Science**: Study how different ML models perform
5. **Geography**: Compare solar potential across different regions

## 🔮 Future Enhancements

The app is designed to be extensible. Potential future features:
- Real-time solar panel monitoring integration
- Battery storage optimization
- Grid integration analysis
- Weather forecasting integration
- Mobile app version

---

**Happy Solar Power Prediction! ☀️⚡**