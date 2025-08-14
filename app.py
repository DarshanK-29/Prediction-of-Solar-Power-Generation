import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Import our custom modules
from weather_api import WeatherAPI
from prediction_model import SolarPowerPredictor

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Solar Power Generation Predictor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .weather-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    """Load the prediction model (cached for performance)."""
    try:
        return SolarPowerPredictor()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_weather_data(city_name):
    """Get weather data for a city (cached for 5 minutes)."""
    try:
        weather_api = WeatherAPI()
        return weather_api.get_weather_data(city_name)
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None

def create_power_gauge(power_value, max_power=100):
    """Create a gauge chart for power generation."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=power_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Solar Power Generation (kW)", 'font': {'size': 20}},
        delta={'reference': max_power * 0.7},
        gauge={
            'axis': {'range': [None, max_power]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_power * 0.3], 'color': "lightgray"},
                {'range': [max_power * 0.3, max_power * 0.7], 'color': "yellow"},
                {'range': [max_power * 0.7, max_power], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_power * 0.9
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_weather_radar(weather_data):
    """Create a radar chart for weather conditions."""
    categories = ['Temperature', 'Humidity', 'Cloud Cover', 'Wind Speed', 'Visibility']
    
    # Normalize values for radar chart
    temp_norm = min(100, max(0, (weather_data['temperature'] + 20) * 2))  # -20 to 30°C -> 0-100
    humidity_norm = weather_data['humidity']  # Already 0-100
    cloud_norm = weather_data['cloud_cover']  # Already 0-100
    wind_norm = min(100, weather_data['wind_speed'] * 10)  # 0-10 m/s -> 0-100
    visibility_norm = min(100, weather_data['visibility'] * 10)  # 0-10 km -> 0-100
    
    values = [temp_norm, humidity_norm, cloud_norm, wind_norm, visibility_norm]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current Conditions',
        line_color='rgb(32, 201, 151)',
        fillcolor='rgba(32, 201, 151, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_feature_importance_chart():
    """Create a feature importance chart if available."""
    try:
        importance_df = pd.read_csv('feature_importance.csv')
        top_features = importance_df.head(10)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 Most Important Features',
            color='importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    except:
        return None

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">☀️ Solar Power Generation Predictor</h1>', unsafe_allow_html=True)
    
    # Check if API key is configured
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        st.error("❌ OpenWeatherMap API key not found! Please add your API key to the .env file.")
        st.info("""
        **To set up your API key:**
        1. Get your free API key from [OpenWeatherMap API](https://openweathermap.org/api)
        2. Copy `.env.example` to `.env`
        3. Add your API key: `OPENWEATHER_API_KEY=your_api_key_here`
        4. Restart the application
        """)
        st.stop()
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # City input (simplified)
    city_name = st.sidebar.text_input(
        "🌍 City Name", 
        value="London", 
        help="Enter the city name (e.g., London, New York, Tokyo)"
    )
    
    # Load model
    predictor = load_predictor()
    
    if predictor is None:
        st.error("❌ Model not found! Please train the model first using `python simple_trainer.py`")
        st.stop()
    
    # Main content
    if st.button("🔍 Get Prediction", type="primary"):
        if not city_name.strip():
            st.error("❌ Please enter a city name")
            st.stop()
        
        with st.spinner("🌤️ Fetching weather data..."):
            weather_data = get_weather_data(city_name)
        
        if weather_data is None:
            st.error("❌ Could not fetch weather data. Please check the city name and try again.")
            st.stop()
        
        with st.spinner("🤖 Making prediction..."):
            prediction_result = predictor.predict(weather_data)
        
        # Display results
        st.success("✅ Prediction completed!")
        
        # Create layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Power generation gauge
            st.subheader("⚡ Power Generation Prediction")
            gauge_fig = create_power_gauge(prediction_result['predicted_power'])
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        with col2:
            # Key metrics
            st.subheader("📊 Key Metrics")
            
            st.markdown(f"""
            <div class="prediction-card">
                <h3>{prediction_result['predicted_power']} kW</h3>
                <p>Predicted Power</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>{prediction_result['confidence']:.1%}</h3>
                <p>Confidence</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="weather-card">
                <h3>{prediction_result['model_used']}</h3>
                <p>Model Used</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Weather conditions
        st.subheader("🌤️ Current Weather Conditions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Temperature",
                f"{weather_data['temperature']:.1f}°C",
                delta=f"{weather_data['temperature'] - 20:.1f}°C from optimal"
            )
        
        with col2:
            st.metric(
                "Humidity",
                f"{weather_data['humidity']:.1f}%",
                delta=f"{weather_data['humidity'] - 50:.1f}% from optimal"
            )
        
        with col3:
            st.metric(
                "Cloud Cover",
                f"{weather_data['cloud_cover']:.1f}%",
                delta=f"{weather_data['cloud_cover'] - 30:.1f}% from optimal"
            )
        
        with col4:
            st.metric(
                "Wind Speed",
                f"{weather_data['wind_speed']:.1f} m/s",
                delta=f"{weather_data['wind_speed'] - 5:.1f} m/s from optimal"
            )
        
        # Weather radar chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📡 Weather Conditions Radar")
            radar_fig = create_weather_radar(weather_data)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        with col2:
            st.subheader("📍 Location & Details")
            st.info(f"""
            **City:** {prediction_result['city']}
            **Weather:** {prediction_result['weather_description']}
            **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            **Solar Noon Distance:** {weather_data['solar_noon_distance']:.1f} hours
            """)
        
        # Detailed weather information
        st.subheader("📋 Detailed Weather Information")
        
        weather_cols = st.columns(3)
        
        with weather_cols[0]:
            st.write("**Temperature & Humidity**")
            st.write(f"Temperature: {weather_data['temperature']:.1f}°C")
            st.write(f"Humidity: {weather_data['humidity']:.1f}%")
            st.write(f"Pressure: {weather_data['pressure']:.0f} hPa")
        
        with weather_cols[1]:
            st.write("**Wind Conditions**")
            st.write(f"Wind Speed: {weather_data['wind_speed']:.1f} m/s")
            st.write(f"Wind Direction: {weather_data['wind_direction']:.0f}°")
            st.write(f"Visibility: {weather_data['visibility']:.1f} km")
        
        with weather_cols[2]:
            st.write("**Solar & Atmospheric**")
            st.write(f"Cloud Cover: {weather_data['cloud_cover']:.1f}%")
            st.write(f"Solar Noon Distance: {weather_data['solar_noon_distance']:.1f} hours")
            st.write(f"Time Features: {weather_data['hour_sin']:.3f}, {weather_data['hour_cos']:.3f}")
        
        # Feature importance (if available)
        importance_fig = create_feature_importance_chart()
        if importance_fig:
            st.subheader("🎯 Model Feature Importance")
            st.plotly_chart(importance_fig, use_container_width=True)
        
        # Model information
        with st.expander("🔧 Model Information"):
            model_info = predictor.get_model_info()
            st.write(f"**Model:** {model_info['model_name']}")
            st.write(f"**Features:** {model_info['feature_count']}")
            st.write(f"**Training Date:** {model_info['training_date']}")
            st.write(f"**Model Path:** {model_info['model_path']}")
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info("""
    This application predicts solar power generation based on real-time weather data.
    
    **Features:**
    - Real-time weather data from OpenWeatherMap
    - Machine learning-based predictions
    - Interactive visualizations
    - Confidence scoring
    
    **How it works:**
    1. Enter a city name
    2. Click "Get Prediction"
    3. View results and visualizations
    """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Built with:**
    - Streamlit
    - Scikit-learn
    - Plotly
    - OpenWeatherMap API
    """)

if __name__ == "__main__":
    main()