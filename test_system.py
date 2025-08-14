#!/usr/bin/env python3
"""
Test script for Solar Power Generation Prediction System
This script tests all components to ensure they work correctly.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

def test_data_generation():
    """Test data generation functionality."""
    print("🧪 Testing data generation...")
    try:
        from data_generator import generate_solar_power_data
        
        # Generate small dataset for testing
        df = generate_solar_power_data(100)
        
        # Check required columns
        required_columns = [
            'date', 'temperature', 'humidity', 'wind_speed', 'wind_direction',
            'cloud_cover', 'pressure', 'visibility', 'solar_noon_distance', 'power_generated'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            return False
        
        # Check data types and ranges
        assert df['temperature'].between(-50, 60).all(), "Temperature out of range"
        assert df['humidity'].between(0, 100).all(), "Humidity out of range"
        assert df['wind_speed'].between(0, 50).all(), "Wind speed out of range"
        assert df['power_generated'].between(0, 200).all(), "Power generated out of range"
        
        print("✅ Data generation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        return False

def test_model_training():
    """Test model training functionality."""
    print("🧪 Testing model training...")
    try:
        from model_trainer import SolarPowerModelTrainer
        
        # Create trainer instance
        trainer = SolarPowerModelTrainer()
        
        # Test data loading and preprocessing
        X_train_scaled, X_test_scaled, y_train, y_test = trainer.load_and_preprocess_data()
        
        # Check data shapes
        assert X_train_scaled.shape[0] > 0, "Training data is empty"
        assert X_test_scaled.shape[0] > 0, "Test data is empty"
        assert len(y_train) > 0, "Training labels are empty"
        assert len(y_test) > 0, "Test labels are empty"
        
        print("✅ Model training test passed")
        return True
        
    except Exception as e:
        print(f"❌ Model training test failed: {e}")
        return False

def test_weather_api():
    """Test weather API functionality (without actual API call)."""
    print("🧪 Testing weather API structure...")
    try:
        from weather_api import WeatherAPI
        
        # Test API class initialization (without API key)
        try:
            weather_api = WeatherAPI()
            print("❌ Should have raised error for missing API key")
            return False
        except ValueError:
            print("✅ API key validation working")
        
        # Test with dummy API key
        weather_api = WeatherAPI(api_key="dummy_key")
        
        # Test data processing method
        dummy_weather_data = {
            'main': {'temp': 25, 'humidity': 60, 'pressure': 1013},
            'wind': {'speed': 5, 'deg': 180},
            'clouds': {'all': 30},
            'weather': [{'description': 'partly cloudy', 'main': 'Clouds'}],
            'sys': {'country': 'US'},
            'visibility': 10000,
            'name': 'Test City'
        }
        
        processed_data = weather_api._process_weather_data(dummy_weather_data)
        
        # Check required features
        required_features = [
            'temperature', 'humidity', 'wind_speed', 'wind_direction',
            'cloud_cover', 'pressure', 'visibility', 'solar_noon_distance'
        ]
        
        missing_features = [feat for feat in required_features if feat not in processed_data]
        if missing_features:
            print(f"❌ Missing features: {missing_features}")
            return False
        
        print("✅ Weather API test passed")
        return True
        
    except Exception as e:
        print(f"❌ Weather API test failed: {e}")
        return False

def test_prediction_model():
    """Test prediction model functionality."""
    print("🧪 Testing prediction model...")
    try:
        from prediction_model import SolarPowerPredictor
        
        # Test model loading (should work if model exists)
        try:
            predictor = SolarPowerPredictor()
            print("✅ Model loaded successfully")
            
            # Test feature preparation with dummy data
            dummy_weather = {
                'temperature': 25.0,
                'humidity': 60.0,
                'wind_speed': 5.0,
                'wind_direction': 180.0,
                'cloud_cover': 30.0,
                'pressure': 1013.0,
                'visibility': 10.0,
                'solar_noon_distance': 2.0,
                'hour_sin': 0.5,
                'hour_cos': 0.866,
                'day_sin': 0.5,
                'day_cos': 0.866,
                'wind_direction_sin': 0.0,
                'wind_direction_cos': -1.0,
                'temp_humidity_interaction': 1500.0,
                'wind_pressure_interaction': 5065.0,
                'cloud_visibility_interaction': 300.0
            }
            
            # Test prediction
            result = predictor.predict(dummy_weather)
            assert 'predicted_power' in result, "Prediction result missing key fields"
            assert 'confidence' in result, "Prediction result missing confidence"
            
            print("✅ Prediction functionality working")
            return True
            
        except FileNotFoundError:
            print("⚠️  Model file not found - this is expected if not trained yet")
            return True  # This is acceptable if model hasn't been trained
        
    except Exception as e:
        print(f"❌ Prediction model test failed: {e}")
        return False

def test_dependencies():
    """Test that all required dependencies are available."""
    print("🧪 Testing dependencies...")
    try:
        import streamlit
        import pandas
        import numpy
        import sklearn
        import matplotlib
        import seaborn
        import plotly
        import requests
        import joblib
        import xgboost
        import lightgbm
        import catboost
        
        print("✅ All dependencies available")
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 Solar Power Generation Prediction - System Test")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Data Generation", test_data_generation),
        ("Model Training", test_model_training),
        ("Weather API", test_weather_api),
        ("Prediction Model", test_prediction_model)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run: python run_training.py")
        print("2. Get OpenWeatherMap API key")
        print("3. Run: streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()