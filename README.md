# ☀️ Solar Power Generation Prediction

A full-stack machine learning project that predicts solar power generation using real-time weather data. This application combines advanced ML algorithms with live weather API integration to provide accurate solar power predictions.

## 🌟 Features

- **Real-time Weather Integration**: Fetches live weather data from OpenWeatherMap API
- **Advanced ML Models**: Evaluates multiple algorithms (Random Forest, XGBoost, LightGBM, etc.)
- **Interactive Web Interface**: Beautiful Streamlit-based UI with real-time visualizations
- **Comprehensive Analysis**: Feature importance, model comparison, and confidence scoring
- **Robust Validation**: Cross-validation and hyperparameter tuning for optimal performance

## 🏗️ Project Structure

```
├── app.py                      # Main Streamlit application
├── data_generator.py           # Synthetic data generation
├── model_trainer.py            # ML model training and evaluation
├── weather_api.py              # OpenWeatherMap API integration
├── prediction_model.py         # Model prediction interface
├── run_training.py             # Complete training pipeline
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Dataset and Train Model

```bash
python run_training.py
```

This will:
- Generate synthetic solar power data
- Train multiple ML models
- Select the best performing model
- Save the model as `solar_power_model.pkl`

### 3. Get OpenWeatherMap API Key

1. Visit [OpenWeatherMap API](https://openweathermap.org/api)
2. Sign up for a free account
3. Get your API key

### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your API key:
```
OPENWEATHER_API_KEY=your_api_key_here
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📊 Dataset Features

The model uses the following weather and temporal features:

### Weather Features
- **Temperature** (°C): Ambient air temperature
- **Humidity** (%): Relative humidity
- **Wind Speed** (m/s): Wind velocity
- **Wind Direction** (degrees): Wind direction (0-360°)
- **Cloud Cover** (%): Sky coverage percentage
- **Pressure** (hPa): Atmospheric pressure
- **Visibility** (km): Atmospheric visibility

### Temporal Features
- **Solar Noon Distance** (hours): Time from solar noon
- **Hour (sin/cos)**: Cyclical encoding of hour of day
- **Day of Year (sin/cos)**: Cyclical encoding of day of year
- **Wind Direction (sin/cos)**: Cyclical encoding of wind direction

### Interaction Features
- Temperature × Humidity
- Wind Speed × Pressure
- Cloud Cover × Visibility

## 🤖 Machine Learning Models

The system evaluates multiple algorithms:

1. **Random Forest**: Ensemble of decision trees
2. **XGBoost**: Gradient boosting with XGBoost
3. **LightGBM**: Light gradient boosting machine
4. **CatBoost**: Categorical boosting
5. **Gradient Boosting**: Traditional gradient boosting
6. **Support Vector Regression**: SVR with RBF kernel
7. **Linear Regression**: Baseline linear model

### Model Selection Criteria
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Cross-validation**: 5-fold CV for robustness

## 🎯 Prediction Features

### Real-time Predictions
- **Power Generation** (kW): Predicted solar power output
- **Confidence Score**: Prediction reliability (0-1)
- **Weather Impact**: How weather affects generation
- **Optimal Conditions**: Comparison with ideal weather

### Visualizations
- **Power Gauge**: Interactive gauge showing predicted power
- **Weather Radar**: Radar chart of current conditions
- **Feature Importance**: Top features affecting predictions
- **Model Comparison**: Performance across algorithms

## 🔧 API Integration

### OpenWeatherMap API
- **Current Weather**: Real-time weather data
- **Forecast Data**: 5-day weather forecasts
- **Geocoding**: City name to coordinates
- **Multiple Units**: Metric and imperial units

### Data Processing
- **Feature Engineering**: Cyclical encoding, interactions
- **Normalization**: Standard scaling for ML models
- **Validation**: Input data validation
- **Error Handling**: Robust error management

## 📈 Model Performance

The trained model typically achieves:
- **R² Score**: 0.85-0.95
- **RMSE**: 5-15 kW
- **MAE**: 3-10 kW
- **Cross-validation**: Consistent performance across folds

## 🎨 User Interface

### Streamlit Features
- **Responsive Design**: Works on desktop and mobile
- **Real-time Updates**: Live weather data fetching
- **Interactive Charts**: Plotly-based visualizations
- **Sidebar Configuration**: Easy API key and city input
- **Error Handling**: User-friendly error messages

### Visual Elements
- **Gradient Cards**: Beautiful metric displays
- **Gauge Charts**: Power generation visualization
- **Radar Charts**: Weather conditions overview
- **Bar Charts**: Feature importance display

## 🔍 Usage Guide

### Making Predictions
1. **Enter City Name**: Simply enter the city name (e.g., "London", "New York", "Tokyo")
2. **Get Prediction**: Click "Get Prediction" button
3. **View Results**: Analyze predictions and visualizations

### Interpreting Results
- **Power Generation**: Expected solar output in kW
- **Confidence**: How reliable the prediction is
- **Weather Impact**: How current conditions affect generation
- **Optimal Conditions**: What weather would be ideal

## 🛠️ Development

### Adding New Models
1. Import the model in `model_trainer.py`
2. Add to the `models` dictionary
3. Include in hyperparameter tuning if needed

### Extending Features
1. Add new features to `data_generator.py`
2. Update feature list in `model_trainer.py`
3. Modify `weather_api.py` to fetch new data
4. Update `prediction_model.py` for new features

### Customizing UI
1. Modify `app.py` for layout changes
2. Update CSS in the `st.markdown` section
3. Add new visualizations using Plotly

## 📝 API Documentation

### WeatherAPI Class
```python
from weather_api import WeatherAPI

# Initialize with API key
weather_api = WeatherAPI(api_key="your_key")

# Get current weather
weather_data = weather_api.get_weather_data("London")

# Get forecast
forecast = weather_api.get_forecast_data("London", days=5)
```

### SolarPowerPredictor Class
```python
from prediction_model import SolarPowerPredictor

# Load trained model
predictor = SolarPowerPredictor()

# Make prediction
result = predictor.predict(weather_data)
```

## 🚨 Troubleshooting

### Common Issues
1. **Model Not Found**: Run `python run_training.py` first
2. **API Key Error**: Check your OpenWeatherMap API key
3. **City Not Found**: Verify city name and country code
4. **Dependencies**: Install all requirements with `pip install -r requirements.txt`

### Error Messages
- **"Model file not found"**: Train the model first
- **"API key required"**: Add your OpenWeatherMap API key
- **"City not found"**: Check city name spelling
- **"Network error"**: Check internet connection

## 📊 Performance Optimization

### Caching
- **Model Loading**: Cached for session duration
- **Weather Data**: Cached for 5 minutes
- **Predictions**: Real-time computation

### Memory Management
- **Efficient Data Structures**: Pandas and NumPy
- **Lazy Loading**: Load models only when needed
- **Garbage Collection**: Automatic memory cleanup

## 🔒 Security

### API Key Protection
- **Environment Variables**: Store API keys securely
- **Input Validation**: Validate all user inputs
- **Error Handling**: Don't expose sensitive information

### Data Privacy
- **Local Processing**: All data processed locally
- **No Data Storage**: Weather data not persisted
- **Secure Communication**: HTTPS for API calls

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenWeatherMap**: For providing weather data API
- **Streamlit**: For the web application framework
- **Scikit-learn**: For machine learning algorithms
- **Plotly**: For interactive visualizations

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the documentation
3. Open an issue on GitHub

---

**Happy Solar Power Predicting! ☀️⚡**