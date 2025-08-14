#!/usr/bin/env python3
"""
Test script to verify realistic solar power predictions.
"""

import os
from dotenv import load_dotenv
from prediction_model import SolarPowerPredictor

# Load environment variables
load_dotenv()

def test_prediction():
    """Test the prediction model with sample weather data."""
    
    print("🧪 Testing Solar Power Prediction Model")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists('solar_power_model.pkl'):
        print("❌ Model file not found! Please run 'python simple_trainer.py' first.")
        return
    
    # Load the model
    try:
        predictor = SolarPowerPredictor()
        print("✅ Model loaded successfully")
        print(f"   Model: {predictor.model_name}")
        print(f"   Features: {len(predictor.feature_names)}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test with different weather scenarios
    test_scenarios = [
        {
            'name': 'Perfect Sunny Day',
            'data': {
                'temperature': 25.0,
                'humidity': 40.0,
                'wind_speed': 3.0,
                'wind_direction': 180.0,
                'cloud_cover': 10.0,
                'pressure': 1013.0,
                'visibility': 10.0,
                'solar_noon_distance': 1.0,
                'hour_sin': 0.5,
                'hour_cos': 0.866,
                'day_sin': 0.5,
                'day_cos': 0.866,
                'wind_direction_sin': 0.0,
                'wind_direction_cos': -1.0,
                'temp_humidity_interaction': 1000.0,
                'wind_pressure_interaction': 3039.0,
                'cloud_visibility_interaction': 100.0,
                'city_name': 'Test City',
                'weather_description': 'Clear sky'
            }
        },
        {
            'name': 'Cloudy Day',
            'data': {
                'temperature': 20.0,
                'humidity': 70.0,
                'wind_speed': 8.0,
                'wind_direction': 270.0,
                'cloud_cover': 80.0,
                'pressure': 1005.0,
                'visibility': 5.0,
                'solar_noon_distance': 3.0,
                'hour_sin': 0.866,
                'hour_cos': 0.5,
                'day_sin': 0.5,
                'day_cos': 0.866,
                'wind_direction_sin': -1.0,
                'wind_direction_cos': 0.0,
                'temp_humidity_interaction': 1400.0,
                'wind_pressure_interaction': 8040.0,
                'cloud_visibility_interaction': 400.0,
                'city_name': 'Test City',
                'weather_description': 'Overcast'
            }
        },
        {
            'name': 'Early Morning',
            'data': {
                'temperature': 15.0,
                'humidity': 85.0,
                'wind_speed': 2.0,
                'wind_direction': 90.0,
                'cloud_cover': 30.0,
                'pressure': 1020.0,
                'visibility': 8.0,
                'solar_noon_distance': 5.0,
                'hour_sin': 0.866,
                'hour_cos': 0.5,
                'day_sin': 0.5,
                'day_cos': 0.866,
                'wind_direction_sin': 1.0,
                'wind_direction_cos': 0.0,
                'temp_humidity_interaction': 1275.0,
                'wind_pressure_interaction': 2040.0,
                'cloud_visibility_interaction': 240.0,
                'city_name': 'Test City',
                'weather_description': 'Partly cloudy'
            }
        }
    ]
    
    print("\n📊 Testing Different Weather Scenarios:")
    print("-" * 50)
    
    for scenario in test_scenarios:
        try:
            result = predictor.predict(scenario['data'])
            
            print(f"\n🌤️ {scenario['name']}:")
            print(f"   Predicted Power: {result['predicted_power']} kW")
            print(f"   Confidence: {result['confidence']:.1%}")
            print(f"   Weather: {result['weather_description']}")
            
            # Validate the prediction
            if result['predicted_power'] > 0:
                print(f"   ✅ Realistic prediction (> 0 kW)")
            else:
                print(f"   ⚠️  Zero prediction (might be night time)")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Prediction test completed!")
    
    # Show model info
    model_info = predictor.get_model_info()
    print(f"\n📋 Model Information:")
    print(f"   Model: {model_info['model_name']}")
    print(f"   Training Date: {model_info['training_date']}")
    print(f"   Feature Count: {model_info['feature_count']}")

if __name__ == "__main__":
    test_prediction()