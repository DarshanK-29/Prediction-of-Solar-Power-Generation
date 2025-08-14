# 🚀 Solar Power Generation Prediction - Setup Guide

This guide will help you set up and run the complete solar power generation prediction system.

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Internet connection for API access

## 🛠️ Installation Steps

### 1. Clone or Download the Project

Make sure you have all the project files in your workspace:
- `app.py` - Main Streamlit application
- `data_generator.py` - Synthetic data generation
- `model_trainer.py` - ML model training
- `weather_api.py` - OpenWeatherMap API integration
- `prediction_model.py` - Model prediction interface
- `requirements.txt` - Python dependencies
- And all other supporting files

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv solar_env

# Activate virtual environment
# On Windows:
solar_env\Scripts\activate
# On macOS/Linux:
source solar_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate Dataset and Train Model

```bash
# Option 1: Use the simplified trainer (recommended)
python simple_trainer.py

# Option 2: Use the complete training pipeline
python run_training.py
```

This will:
- Generate synthetic solar power data (`augmented_solarpowergeneration.csv`)
- Train multiple ML models
- Select the best performing model
- Save the model as `solar_power_model.pkl`
- Generate evaluation plots and feature importance

### 5. Get OpenWeatherMap API Key

1. Visit [OpenWeatherMap API](https://openweathermap.org/api)
2. Sign up for a free account
3. Get your API key from the dashboard
4. Copy the API key for use in the application

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 🎯 Using the Application

### 1. Enter City Name
- In the sidebar, simply enter a city name (e.g., "London", "New York", "Tokyo")
- The API key is automatically loaded from the environment file

### 2. Get Prediction
- Click the "Get Prediction" button
- The system will:
  - Fetch current weather data for the city
  - Process the weather features
  - Make a solar power generation prediction
  - Display results with visualizations

### 3. Interpret Results
- **Power Generation**: Predicted solar output in kW
- **Confidence**: How reliable the prediction is (0-1)
- **Weather Impact**: How current conditions affect generation
- **Visualizations**: Interactive charts showing predictions and weather

## 📊 Understanding the Results

### Power Generation Prediction
- Shows expected solar power output based on current weather
- Values typically range from 0-100 kW depending on conditions
- Higher values indicate better solar generation potential

### Confidence Score
- Indicates prediction reliability
- Factors affecting confidence:
  - Weather conditions (cloud cover, temperature)
  - Time of day (distance from solar noon)
  - Wind conditions
  - Data quality

### Weather Conditions
- **Temperature**: Optimal around 25°C
- **Humidity**: Lower is generally better
- **Cloud Cover**: Major factor affecting solar generation
- **Wind Speed**: Can affect panel efficiency
- **Solar Noon Distance**: Time from peak sun hours

## 🔧 Troubleshooting

### Common Issues

1. **"Model file not found"**
   - Solution: Run `python simple_trainer.py` first

2. **"API key not found"**
   - Solution: Add your OpenWeatherMap API key to the `.env` file

3. **"City not found"**
   - Solution: Check city name spelling and try a different city

4. **"Dependencies not found"**
   - Solution: Run `pip install -r requirements.txt`

5. **"Port already in use"**
   - Solution: Use a different port: `streamlit run app.py --server.port 8502`

### Error Messages

- **Network errors**: Check internet connection
- **API rate limits**: Free OpenWeatherMap API has limits
- **Memory issues**: Close other applications to free up RAM

## 📈 Model Performance

The trained model typically achieves:
- **R² Score**: 0.85-0.95 (very high accuracy)
- **RMSE**: 5-15 kW
- **MAE**: 3-10 kW
- **Cross-validation**: Consistent performance

## 🎨 Features

### Real-time Predictions
- Live weather data integration
- Instant solar power predictions
- Confidence scoring
- Weather impact analysis

### Visualizations
- Interactive power gauge
- Weather radar charts
- Feature importance plots
- Model comparison charts

### Data Processing
- 17 engineered features
- Cyclical encoding for time/direction
- Interaction features
- Robust scaling and validation

## 🔒 Security & Privacy

- API keys are stored locally only
- No data is transmitted except to OpenWeatherMap
- All processing happens locally
- No personal data is collected

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Ensure you have a valid API key
4. Check the console for error messages
5. Restart the application if needed

## 🎉 Success!

Once everything is working, you should see:
- A beautiful Streamlit interface
- Real-time weather data fetching
- Accurate solar power predictions
- Interactive visualizations
- Confidence scores and analysis

**Happy Solar Power Predicting! ☀️⚡**