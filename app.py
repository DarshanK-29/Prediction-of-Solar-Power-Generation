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

# Import advanced features
try:
    from ensemble_predictor import AdvancedEnsemblePredictor
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

try:
    from roi_calculator import SolarROICalculator
    ROI_AVAILABLE = True
except ImportError:
    ROI_AVAILABLE = False

try:
    from solar_3d_visualizer import Solar3DVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False

try:
    from satellite_data_integration import SatelliteDataIntegration
    SATELLITE_AVAILABLE = True
except ImportError:
    SATELLITE_AVAILABLE = False

try:
    from carbon_footprint_tracker import CarbonFootprintTracker
    CARBON_AVAILABLE = True
except ImportError:
    CARBON_AVAILABLE = False

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Advanced Solar Power Generation Predictor",
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
    .advanced-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stAlert {
        border-radius: 10px;
    }
    .feature-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
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

def show_basic_prediction(city_name, api_key):
    """Show basic prediction functionality."""
    st.subheader("🔍 Basic Solar Power Prediction")
    
    # Load model
    predictor = load_predictor()
    
    if predictor is None:
        st.error("❌ Model not found! Please train the model first using `python simple_trainer.py`")
        return
    
    if st.button("🔍 Get Basic Prediction", type="primary"):
        if not city_name.strip():
            st.error("❌ Please enter a city name")
            return
        
        with st.spinner("🌤️ Fetching weather data..."):
            weather_data = get_weather_data(city_name)
        
        if weather_data is None:
            st.error("❌ Could not fetch weather data. Please check the city name and try again.")
            return
        
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
        
        # Model information
        with st.expander("🔧 Model Information"):
            model_info = predictor.get_model_info()
            st.write(f"**Model:** {model_info['model_name']}")
            st.write(f"**Features:** {model_info['feature_count']}")
            st.write(f"**Training Date:** {model_info['training_date']}")

def show_ensemble_prediction(city_name, api_key):
    """Show ensemble prediction functionality."""
    st.subheader("🤖 Advanced Ensemble ML Prediction")
    st.info("This feature uses multiple ML models and dynamically selects the best one based on weather conditions.")
    
    try:
        ensemble_predictor = AdvancedEnsemblePredictor()
        
        if st.button("🤖 Get Ensemble Prediction", type="primary"):
            if not city_name.strip():
                st.error("❌ Please enter a city name")
                return
            
            with st.spinner("🌤️ Fetching weather data..."):
                weather_data = get_weather_data(city_name)
            
            if weather_data is None:
                st.error("❌ Could not fetch weather data. Please check the city name and try again.")
                return
            
            with st.spinner("🧠 Making ensemble prediction..."):
                result = ensemble_predictor.predict(weather_data)
            
            st.success("✅ Ensemble prediction completed!")
            
            # Display ensemble results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("⚡ Ensemble Power Generation")
                gauge_fig = create_power_gauge(result['predicted_power'])
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 Ensemble Metrics")
                
                st.markdown(f"""
                <div class="advanced-card">
                    <h3>{result['predicted_power']} kW</h3>
                    <p>Ensemble Prediction</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{result['confidence']:.1%}</h3>
                <p>Confidence</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="weather-card">
                    <h3>{result['selected_model']}</h3>
                    <p>Selected Model</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show all model predictions
            st.subheader("🤖 All Model Predictions")
            model_predictions = result['all_model_predictions']
            
            col1, col2, col3, col4, col5 = st.columns(5)
            for i, (model_name, prediction) in enumerate(model_predictions.items()):
                with [col1, col2, col3, col4, col5][i]:
                    st.metric(model_name, f"{prediction} kW")
            
            # Ensemble info
            with st.expander("🔧 Ensemble Information"):
                st.write(f"**Total Models:** {result['ensemble_info']['total_models']}")
                st.write(f"**Model Names:** {', '.join(result['ensemble_info']['model_names'])}")
                st.write(f"**Meta-Learner:** {result['ensemble_info']['meta_learner_type']}")
                
    except Exception as e:
        st.error(f"❌ Error with ensemble prediction: {e}")
        st.info("Please ensure ensemble models are trained using `python ensemble_trainer.py`")

def show_roi_calculator():
    """Show ROI calculator functionality."""
    st.subheader("💰 Solar Installation ROI Calculator")
    st.info("Calculate return on investment, payback period, and financial benefits for solar installations.")
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        system_size = st.number_input("System Size (kW)", min_value=1.0, max_value=100.0, value=5.0, step=0.5)
        daily_generation = st.number_input("Daily Generation (kWh)", min_value=1.0, max_value=100.0, value=20.0, step=0.5)
        region = st.selectbox("Region", [
            'US_National', 'California', 'Texas', 'New_York', 'Florida', 'Arizona',
            'Europe_UK', 'Europe_Germany', 'Asia_China', 'Asia_Japan', 'Australia'
        ])
    
    with col2:
        years = st.slider("Analysis Period (years)", min_value=10, max_value=30, value=25)
        custom_installation_cost = st.number_input("Custom Installation Cost ($/kW)", min_value=1000, max_value=5000, value=2500, step=100)
    
    if st.button("💰 Calculate ROI", type="primary"):
        try:
            calculator = SolarROICalculator()
            
            # Custom parameters
            custom_params = {
                'installation_cost_per_kw': custom_installation_cost
            }
            
            roi_result = calculator.calculate_roi(
                system_size_kw=system_size,
                daily_generation_kwh=daily_generation,
                region=region,
                custom_params=custom_params,
                years=years
            )
            
            st.success("✅ ROI calculation completed!")
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("💰 Financial Metrics")
                st.metric("ROI", f"{roi_result['financial_metrics']['roi_percent']:.1f}%")
                st.metric("Payback Period", f"{roi_result['financial_metrics']['payback_period_years']} years")
                st.metric("Total Savings", f"${roi_result['financial_metrics']['total_savings']:,.0f}")
                st.metric("NPV", f"${roi_result['financial_metrics']['npv']:,.0f}")
            
            with col2:
                st.subheader("🌱 Environmental Impact")
                st.metric("CO2 Saved", f"{roi_result['environmental_benefits']['co2_saved_kg']:,.0f} kg")
                st.metric("Trees Equivalent", f"{roi_result['environmental_benefits']['trees_equivalent']:,.0f}")
                st.metric("Cars Off Road", f"{roi_result['environmental_benefits']['cars_off_road']:.1f}")
            
            with col3:
                st.subheader("📊 System Information")
                st.metric("Installation Cost", f"${roi_result['costs']['net_installation_cost']:,.0f}")
                st.metric("Annual Generation", f"{roi_result['system_info']['annual_generation_kwh']:,.0f} kWh")
                st.metric("Electricity Price", f"${roi_result['system_info']['electricity_price_usd_per_kwh']:.2f}/kWh")
            
            # Annual breakdown
            st.subheader("📈 Annual Cash Flow Breakdown")
            annual_data = roi_result['annual_breakdown']
            df = pd.DataFrame(annual_data)
            st.dataframe(df, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error calculating ROI: {e}")

def show_3d_visualizations(city_name):
    """Show 3D visualization functionality."""
    st.subheader("📊 3D Solar Irradiance Visualizations")
    st.info("Interactive 3D visualizations showing solar potential across different times and conditions.")
    
    # Get coordinates for the city (simplified - you could use a geocoding API)
    city_coords = {
        'London': (51.5074, -0.1278),
        'New York': (40.7128, -74.0060),
        'Tokyo': (35.6762, 139.6503),
        'Paris': (48.8566, 2.3522),
        'Sydney': (-33.8688, 151.2093),
        'Mumbai': (19.0760, 72.8777),
        'Beijing': (39.9042, 116.4074),
        'Berlin': (52.5200, 13.4050),
        'Rome': (41.9028, 12.4964),
        'Madrid': (40.4168, -3.7038)
    }
    
    lat, lon = city_coords.get(city_name, (40.7128, -74.0060))  # Default to New York
    
    try:
        visualizer = Solar3DVisualizer()
        
        # Visualization options
        viz_type = st.selectbox("Choose Visualization", [
            "Daily Irradiance Profile",
            "Weather Impact Analysis", 
            "Solar Path Visualization",
            "Seasonal Comparison"
        ])
        
        if st.button("📊 Generate Visualization", type="primary"):
            with st.spinner("Creating visualization..."):
                if viz_type == "Daily Irradiance Profile":
                    fig = visualizer.create_daily_irradiance_profile(lat, lon)
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif viz_type == "Weather Impact Analysis":
                    fig = visualizer.create_weather_impact_analysis(lat, lon)
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif viz_type == "Solar Path Visualization":
                    fig = visualizer.create_solar_path_visualization(lat, lon)
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif viz_type == "Seasonal Comparison":
                    fig = visualizer.create_seasonal_comparison(lat, lon)
                    st.plotly_chart(fig, use_container_width=True)
            
            st.success("✅ Visualization created successfully!")
            
    except Exception as e:
        st.error(f"❌ Error creating visualization: {e}")

def show_satellite_data(city_name):
    """Show satellite data functionality."""
    st.subheader("🛰️ Satellite Data Integration")
    st.info("Enhanced weather data combining satellite imagery with ground-based weather stations.")
    
    # Get coordinates for the city (simplified)
    city_coords = {
        'London': (51.5074, -0.1278),
        'New York': (40.7128, -74.0060),
        'Tokyo': (35.6762, 139.6503),
        'Paris': (48.8566, 2.3522),
        'Sydney': (-33.8688, 151.2093),
        'Mumbai': (19.0760, 72.8777),
        'Beijing': (39.9042, 116.4074),
        'Berlin': (52.5200, 13.4050),
        'Rome': (41.9028, 12.4964),
        'Madrid': (40.4168, -3.7038)
    }
    
    lat, lon = city_coords.get(city_name, (40.7128, -74.0060))
    
    try:
        satellite = SatelliteDataIntegration()
        
        if st.button("🛰️ Get Enhanced Weather Data", type="primary"):
            with st.spinner("Fetching satellite data..."):
                enhanced_data = satellite.get_enhanced_weather_data(lat, lon, city_name)
            
            st.success("✅ Enhanced weather data retrieved!")
            
            # Display enhanced data
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🌤️ Enhanced Weather Data")
                st.metric("Cloud Cover", f"{enhanced_data['cloud_cover']}%")
                st.metric("Temperature", f"{enhanced_data['temperature']}°C")
                st.metric("Humidity", f"{enhanced_data['humidity']}%")
                st.metric("Wind Speed", f"{enhanced_data['wind_speed']} m/s")
                st.metric("Confidence", f"{enhanced_data['confidence']:.1%}")
            
            with col2:
                st.subheader("🛰️ Satellite Information")
                st.metric("Data Sources", f"{len(enhanced_data['data_sources'])} sources")
                st.metric("Satellite Resolution", enhanced_data['satellite_details']['resolution'])
                st.metric("Cloud Type", enhanced_data['satellite_details']['cloud_type'])
                st.metric("Cloud Height", f"{enhanced_data['satellite_details']['cloud_height_km']:.1f} km")
            
            # Data sources
            st.subheader("📡 Data Sources")
            for source_type, source_name in enhanced_data['data_sources'].items():
                st.info(f"**{source_type.title()}:** {source_name}")
            
            # Cloud trend analysis
            st.subheader("📈 Cloud Trend Analysis")
            if st.button("📊 Analyze Cloud Trends"):
                with st.spinner("Analyzing cloud trends..."):
                    trend_analysis = satellite.analyze_cloud_trends(lat, lon, days=30)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Trend Direction", trend_analysis['trend_direction'])
                    st.metric("Mean Cloud Cover", f"{trend_analysis['mean_cloud_cover']}%")
                
                with col2:
                    st.metric("Data Points", trend_analysis['data_points'])
                    st.metric("Confidence", f"{trend_analysis['confidence']:.1%}")
                
                with col3:
                    st.metric("Period (days)", trend_analysis['period_days'])
                    st.metric("Trend Magnitude", trend_analysis['trend_magnitude'])
                
    except Exception as e:
        st.error(f"❌ Error with satellite data: {e}")

def show_carbon_tracker():
    """Show carbon tracker functionality."""
    st.subheader("🌱 Carbon Footprint Reduction Tracker")
    st.info("Track CO2 emissions saved and environmental impact of solar generation.")
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        solar_generation = st.number_input("Solar Generation (kWh)", min_value=1.0, max_value=1000.0, value=20.0, step=0.5)
        region = st.selectbox("Region", [
            'US_National', 'California', 'Texas', 'New_York', 'Florida', 'Arizona',
            'Europe_UK', 'Europe_Germany', 'Asia_China', 'Asia_Japan', 'Australia'
        ])
        time_period = st.selectbox("Time Period", ['daily', 'monthly', 'yearly'])
    
    with col2:
        system_size = st.number_input("System Size (kW)", min_value=1.0, max_value=100.0, value=5.0, step=0.5)
        tracking_days = st.slider("Tracking Period (days)", min_value=7, max_value=365, value=30)
    
    if st.button("🌱 Calculate Carbon Impact", type="primary"):
        try:
            tracker = CarbonFootprintTracker()
            
            # Calculate carbon savings
            carbon_savings = tracker.calculate_carbon_savings(
                solar_generation_kwh=solar_generation,
                region=region,
                time_period=time_period
            )
            
            st.success("✅ Carbon impact calculated!")
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("🌱 Carbon Savings")
                st.metric("CO2 Saved", f"{carbon_savings['co2_saved_kg']:.1f} kg")
                st.metric("Trees Equivalent", f"{carbon_savings['environmental_impact']['trees_planted']:.1f}")
                st.metric("Cars Off Road", f"{carbon_savings['environmental_impact']['cars_off_road']:.2f}")
            
            with col2:
                st.subheader("📊 Environmental Equivalencies")
                st.metric("Smartphones Charged", f"{carbon_savings['environmental_impact']['smartphones_charged']:.0f}")
                st.metric("Lightbulb Hours", f"{carbon_savings['environmental_impact']['lightbulb_hours']:.0f}")
                st.metric("Flight Kilometers", f"{carbon_savings['environmental_impact']['flight_km']:.1f}")
            
            with col3:
                st.subheader("💰 Financial Impact")
                st.metric("Grid Emission Factor", f"{carbon_savings['grid_emission_factor']:.3f} kg CO2/kWh")
                st.metric("Annual Generation", f"{carbon_savings['time_metrics']['annual_generation_kwh']:,.0f} kWh")
                st.metric("Lifetime CO2 Saved", f"{carbon_savings['time_metrics']['lifetime_co2_saved_kg']:,.0f} kg")
            
            # Regional comparison
            st.subheader("🌍 Regional Impact Comparison")
            if st.button("🌍 Compare Regions"):
                regional_comparison = tracker.compare_regional_impact(solar_generation)
                
                # Create comparison chart
                regions = list(regional_comparison.keys())
                co2_savings = [regional_comparison[region]['co2_saved_kg'] for region in regions]
                
                fig = px.bar(
                    x=regions,
                    y=co2_savings,
                    title="CO2 Savings by Region",
                    labels={'x': 'Region', 'y': 'CO2 Saved (kg)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # System efficiency
            st.subheader("⚡ System Efficiency Analysis")
            if st.button("⚡ Analyze Efficiency"):
                efficiency_metrics = tracker.calculate_system_efficiency_metrics(
                    system_size_kw=system_size,
                    actual_generation_kwh=solar_generation * tracking_days,
                    time_period_days=tracking_days
                )
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Efficiency", f"{efficiency_metrics['efficiency_percent']}%")
                    st.metric("Performance Rating", efficiency_metrics['performance_rating'])
                
                with col2:
                    st.metric("Actual Generation", f"{efficiency_metrics['actual_generation_kwh']:,.0f} kWh")
                    st.metric("Theoretical Generation", f"{efficiency_metrics['theoretical_generation_kwh']:,.0f} kWh")
                
                with col3:
                    st.metric("Efficiency-Adjusted CO2", f"{efficiency_metrics['efficiency_adjusted_co2_kg']:.1f} kg")
                
    except Exception as e:
        st.error(f"❌ Error calculating carbon impact: {e}")

def main():
    """Main application function with advanced features."""
    
    # Header with advanced features badges
    st.markdown('<h1 class="main-header">☀️ Advanced Solar Power Generation Predictor</h1>', unsafe_allow_html=True)
    
    # Display feature badges
    st.markdown("### 🚀 Advanced Features Available")
    features_html = ""
    if ENSEMBLE_AVAILABLE:
        features_html += '<span class="feature-badge">🤖 Ensemble ML</span>'
    if ROI_AVAILABLE:
        features_html += '<span class="feature-badge">💰 ROI Calculator</span>'
    if VISUALIZER_AVAILABLE:
        features_html += '<span class="feature-badge">📊 3D Maps</span>'
    if SATELLITE_AVAILABLE:
        features_html += '<span class="feature-badge">🛰️ Satellite Data</span>'
    if CARBON_AVAILABLE:
        features_html += '<span class="feature-badge">🌱 Carbon Tracker</span>'
    
    st.markdown(features_html, unsafe_allow_html=True)
    st.markdown("---")
    
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
    
    # City input
    city_name = st.sidebar.text_input(
        "🌍 City Name", 
        value="London", 
        help="Enter the city name (e.g., London, New York, Tokyo)"
    )
    
    # Create tabs for different features
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Basic Prediction", 
        "🤖 Ensemble ML", 
        "💰 ROI Calculator", 
        "📊 3D Visualizations", 
        "🛰️ Satellite Data", 
        "🌱 Carbon Tracker"
    ])

    # Tab 1: Basic Prediction
    with tab1:
        show_basic_prediction(city_name, api_key)

    # Tab 2: Ensemble ML
    with tab2:
        if ENSEMBLE_AVAILABLE:
            show_ensemble_prediction(city_name, api_key)
        else:
            st.warning("⚠️ Ensemble ML feature not available. Please ensure ensemble models are trained.")

    # Tab 3: ROI Calculator
    with tab3:
        if ROI_AVAILABLE:
            show_roi_calculator()
        else:
            st.warning("⚠️ ROI Calculator feature not available.")

    # Tab 4: 3D Visualizations
    with tab4:
        if VISUALIZER_AVAILABLE:
            show_3d_visualizations(city_name)
        else:
            st.warning("⚠️ 3D Visualizations feature not available.")

    # Tab 5: Satellite Data
    with tab5:
        if SATELLITE_AVAILABLE:
            show_satellite_data(city_name)
        else:
            st.warning("⚠️ Satellite Data feature not available.")

    # Tab 6: Carbon Tracker
    with tab6:
        if CARBON_AVAILABLE:
            show_carbon_tracker()
        else:
            st.warning("⚠️ Carbon Tracker feature not available.")
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info("""
    **Advanced Solar Power Prediction System**
    
    **Core Features:**
    - Real-time weather data from OpenWeatherMap
    - Machine learning-based predictions
    - Interactive visualizations
    - Confidence scoring
    
    **Advanced Features:**
    - Multi-Model Ensemble with Dynamic Selection
    - ROI Calculator for Solar Installations
    - 3D Solar Irradiance Maps
    - Satellite Data Integration
    - Carbon Footprint Reduction Tracker
    
    **How it works:**
    1. Enter a city name
    2. Choose a feature tab
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
    - NASA APIs
    - Advanced ML Ensemble
    """)

if __name__ == "__main__":
    main()