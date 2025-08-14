#!/usr/bin/env python3
"""
Test script to verify all advanced features in the Solar Power Prediction App
"""

import sys
import os
from datetime import datetime

def test_imports():
    """Test if all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        from weather_api import WeatherAPI
        print("✅ WeatherAPI imported")
    except ImportError as e:
        print(f"❌ WeatherAPI import failed: {e}")
        return False
    
    try:
        from prediction_model import SolarPowerPredictor
        print("✅ SolarPowerPredictor imported")
    except ImportError as e:
        print(f"❌ SolarPowerPredictor import failed: {e}")
        return False
    
    # Test advanced features
    advanced_features = [
        ('ensemble_predictor', 'AdvancedEnsemblePredictor'),
        ('roi_calculator', 'SolarROICalculator'),
        ('solar_3d_visualizer', 'Solar3DVisualizer'),
        ('satellite_data_integration', 'SatelliteDataIntegration'),
        ('carbon_footprint_tracker', 'CarbonFootprintTracker')
    ]
    
    for module_name, class_name in advanced_features:
        try:
            module = __import__(module_name)
            class_obj = getattr(module, class_name)
            print(f"✅ {class_name} imported")
        except ImportError as e:
            print(f"❌ {class_name} import failed: {e}")
            return False
    
    return True

def test_ensemble_models():
    """Test if ensemble models are available."""
    print("\n🤖 Testing ensemble models...")
    
    try:
        from ensemble_predictor import AdvancedEnsemblePredictor
        predictor = AdvancedEnsemblePredictor()
        print("✅ Ensemble predictor loaded")
        
        # Test with sample data
        sample_weather = {
            'temperature': 25.0,
            'humidity': 60.0,
            'wind_speed': 5.0,
            'wind_direction': 180.0,
            'cloud_cover': 30.0,
            'pressure': 1013.0,
            'visibility': 10.0,
            'solar_noon_distance': 2.0
        }
        
        result = predictor.predict(sample_weather)
        print(f"✅ Ensemble prediction successful: {result['predicted_power']:.2f} kW")
        return True
        
    except Exception as e:
        print(f"❌ Ensemble models test failed: {e}")
        return False

def test_roi_calculator():
    """Test ROI calculator functionality."""
    print("\n💰 Testing ROI calculator...")
    
    try:
        from roi_calculator import SolarROICalculator
        calculator = SolarROICalculator()
        
        result = calculator.calculate_roi(
            system_size_kw=5.0,
            daily_generation_kwh=20.0,
            region='US_National'
        )
        
        print(f"✅ ROI calculation successful: {result['financial_metrics']['roi_percent']:.1f}% ROI")
        return True
        
    except Exception as e:
        print(f"❌ ROI calculator test failed: {e}")
        return False

def test_3d_visualizer():
    """Test 3D visualizer functionality."""
    print("\n📊 Testing 3D visualizer...")
    
    try:
        from solar_3d_visualizer import Solar3DVisualizer
        visualizer = Solar3DVisualizer()
        
        # Test daily irradiance profile
        fig = visualizer.create_daily_irradiance_profile(40.7128, -74.0060)  # New York
        print("✅ 3D visualizer created daily irradiance profile")
        return True
        
    except Exception as e:
        print(f"❌ 3D visualizer test failed: {e}")
        return False

def test_satellite_data():
    """Test satellite data integration."""
    print("\n🛰️ Testing satellite data integration...")
    
    try:
        from satellite_data_integration import SatelliteDataIntegration
        satellite = SatelliteDataIntegration()
        
        # Test enhanced weather data
        enhanced_data = satellite.get_enhanced_weather_data(40.7128, -74.0060, "New York")
        print(f"✅ Satellite data integration successful: {enhanced_data['cloud_cover']}% cloud cover")
        return True
        
    except Exception as e:
        print(f"❌ Satellite data test failed: {e}")
        return False

def test_carbon_tracker():
    """Test carbon footprint tracker."""
    print("\n🌱 Testing carbon footprint tracker...")
    
    try:
        from carbon_footprint_tracker import CarbonFootprintTracker
        tracker = CarbonFootprintTracker()
        
        result = tracker.calculate_carbon_savings(
            solar_generation_kwh=20.0,
            region='US_National'
        )
        
        print(f"✅ Carbon tracker successful: {result['co2_saved_kg']:.1f} kg CO2 saved")
        return True
        
    except Exception as e:
        print(f"❌ Carbon tracker test failed: {e}")
        return False

def test_app_structure():
    """Test if the app.py structure is correct."""
    print("\n📱 Testing app structure...")
    
    try:
        # Read app.py and check for key components
        with open('app.py', 'r') as f:
            content = f.read()
        
        required_components = [
            'def show_ensemble_prediction',
            'def show_roi_calculator',
            'def show_3d_visualizations',
            'def show_satellite_data',
            'def show_carbon_tracker',
            'st.tabs',
            'AdvancedEnsemblePredictor',
            'SolarROICalculator',
            'Solar3DVisualizer',
            'SatelliteDataIntegration',
            'CarbonFootprintTracker'
        ]
        
        missing_components = []
        for component in required_components:
            if component not in content:
                missing_components.append(component)
        
        if missing_components:
            print(f"❌ Missing components in app.py: {missing_components}")
            return False
        else:
            print("✅ All required components found in app.py")
            return True
            
    except Exception as e:
        print(f"❌ App structure test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Advanced Solar Power Prediction App Features")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_ensemble_models,
        test_roi_calculator,
        test_3d_visualizer,
        test_satellite_data,
        test_carbon_tracker,
        test_app_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The advanced features are ready to use.")
        print("\n🚀 To run the application:")
        print("   streamlit run app.py")
        print("\n📋 Available features:")
        print("   • Basic Solar Power Prediction")
        print("   • 🤖 Multi-Model Ensemble with Dynamic Selection")
        print("   • 💰 ROI Calculator for Solar Installations")
        print("   • 📊 3D Solar Irradiance Maps")
        print("   • 🛰️ Satellite Data Integration")
        print("   • 🌱 Carbon Footprint Reduction Tracker")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)