#!/usr/bin/env python3
"""
Comprehensive Test Script for Advanced Solar Power Features
Tests all 5 advanced features implemented in the project.
"""

import os
import sys
from datetime import datetime

def test_feature_1_ensemble_model():
    """Test Feature #1: Multi-Model Ensemble with Dynamic Selection"""
    print("🧪 Testing Feature #1: Multi-Model Ensemble with Dynamic Selection")
    print("-" * 60)
    
    try:
        # Check if ensemble model exists
        if os.path.exists('ensemble_models/meta_learner.pkl'):
            print("✅ Ensemble model files found")
            
            # Test ensemble predictor
            from ensemble_predictor import AdvancedEnsemblePredictor
            
            predictor = AdvancedEnsemblePredictor()
            
            # Test weather data
            test_weather = {
                'temperature': 25.0,
                'humidity': 40.0,
                'wind_speed': 3.0,
                'wind_direction': 180.0,
                'cloud_cover': 10.0,
                'pressure': 1013.0,
                'visibility': 10.0,
                'solar_noon_distance': 1.0,
                'city_name': 'Test City',
                'weather_description': 'Clear sky'
            }
            
            result = predictor.predict(test_weather)
            
            print(f"   ✅ Ensemble prediction: {result['predicted_power']} kW")
            print(f"   ✅ Selected model: {result['selected_model']}")
            print(f"   ✅ Confidence: {result['confidence']:.1%}")
            print(f"   ✅ Total models: {result['ensemble_info']['total_models']}")
            
            return True
        else:
            print("⚠️  Ensemble model not found - run 'python ensemble_trainer.py' first")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_feature_2_roi_calculator():
    """Test Feature #2: ROI Calculator for Solar Installations"""
    print("\n🧪 Testing Feature #2: ROI Calculator for Solar Installations")
    print("-" * 60)
    
    try:
        from roi_calculator import SolarROICalculator
        
        calculator = SolarROICalculator()
        
        # Test ROI calculation
        roi_result = calculator.calculate_roi(
            system_size_kw=5.0,
            daily_generation_kwh=20.0,
            region='California',
            years=25
        )
        
        print(f"   ✅ ROI: {roi_result['financial_metrics']['roi_percent']:.1f}%")
        print(f"   ✅ Payback Period: {roi_result['financial_metrics']['payback_period_years']} years")
        print(f"   ✅ Total Savings: ${roi_result['financial_metrics']['total_savings']:,.2f}")
        print(f"   ✅ CO2 Saved: {roi_result['environmental_benefits']['co2_saved_kg']:,.0f} kg")
        
        # Test optimal system size
        optimal_result = calculator.calculate_optimal_system_size(
            daily_consumption_kwh=30.0,
            region='California',
            budget_usd=15000
        )
        
        print(f"   ✅ Recommended System Size: {optimal_result['recommended_system_size_kw']} kW")
        print(f"   ✅ System Options: {len(optimal_result['system_options'])}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_feature_3_3d_visualization():
    """Test Feature #3: 3D Solar Irradiance Maps"""
    print("\n🧪 Testing Feature #3: 3D Solar Irradiance Maps")
    print("-" * 60)
    
    try:
        from solar_3d_visualizer import Solar3DVisualizer
        
        visualizer = Solar3DVisualizer()
        
        # Test coordinates (New York City)
        lat, lon = 40.7128, -74.0060
        
        # Test daily irradiance profile
        daily_fig = visualizer.create_daily_irradiance_profile(lat, lon)
        print("   ✅ Daily irradiance profile created")
        
        # Test weather impact analysis
        weather_fig = visualizer.create_weather_impact_analysis(lat, lon)
        print("   ✅ Weather impact analysis created")
        
        # Test solar path visualization
        solar_path_fig = visualizer.create_solar_path_visualization(lat, lon)
        print("   ✅ Solar path visualization created")
        
        # Test seasonal comparison
        seasonal_fig = visualizer.create_seasonal_comparison(lat, lon)
        print("   ✅ Seasonal comparison created")
        
        print("   ✅ All 3D visualizations created successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_feature_4_satellite_integration():
    """Test Feature #4: Satellite Data Integration"""
    print("\n🧪 Testing Feature #4: Satellite Data Integration")
    print("-" * 60)
    
    try:
        from satellite_data_integration import SatelliteDataIntegration
        
        satellite = SatelliteDataIntegration()
        
        # Test coordinates (New York City)
        lat, lon = 40.7128, -74.0060
        
        # Test enhanced weather data
        enhanced_data = satellite.get_enhanced_weather_data(lat, lon, "New York")
        
        print(f"   ✅ Cloud Cover: {enhanced_data['cloud_cover']}%")
        print(f"   ✅ Temperature: {enhanced_data['temperature']}°C")
        print(f"   ✅ Confidence: {enhanced_data['confidence']}")
        print(f"   ✅ Data Sources: {enhanced_data['data_sources']}")
        print(f"   ✅ Satellite: {enhanced_data['satellite_details']['resolution']}")
        
        # Test cloud trend analysis
        trend_analysis = satellite.analyze_cloud_trends(lat, lon, days=7)
        
        print(f"   ✅ Trend Direction: {trend_analysis['trend_direction']}")
        print(f"   ✅ Mean Cloud Cover: {trend_analysis['mean_cloud_cover']}%")
        print(f"   ✅ Data Points: {trend_analysis['data_points']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_feature_5_carbon_tracker():
    """Test Feature #5: Carbon Footprint Reduction Tracker"""
    print("\n🧪 Testing Feature #5: Carbon Footprint Reduction Tracker")
    print("-" * 60)
    
    try:
        from carbon_footprint_tracker import CarbonFootprintTracker
        
        tracker = CarbonFootprintTracker()
        
        # Test daily carbon savings
        daily_savings = tracker.calculate_carbon_savings(20.0, 'California', 'daily')
        
        print(f"   ✅ Daily Generation: {daily_savings['solar_generation_kwh']} kWh")
        print(f"   ✅ CO2 Saved: {daily_savings['co2_saved_kg']} kg")
        print(f"   ✅ Trees Equivalent: {daily_savings['environmental_impact']['trees_planted']}")
        print(f"   ✅ Cars Equivalent: {daily_savings['environmental_impact']['cars_off_road']}")
        
        # Test tracking over time
        daily_generations = [15, 18, 22, 19, 25, 20, 17]  # 7 days of data
        tracking_data = tracker.track_daily_carbon_savings(daily_generations, 'California')
        
        print(f"   ✅ Total CO2 Saved: {tracking_data['total_co2_saved_kg']} kg")
        print(f"   ✅ Average Daily CO2 Saved: {tracking_data['average_daily_co2_saved_kg']} kg")
        
        # Test regional comparison
        regional_comparison = tracker.compare_regional_impact(20.0)
        print(f"   ✅ Regional comparisons: {len(regional_comparison)} regions")
        
        # Test system efficiency
        efficiency_metrics = tracker.calculate_system_efficiency_metrics(5.0, 140.0, 7)
        print(f"   ✅ System Efficiency: {efficiency_metrics['efficiency_percent']}%")
        print(f"   ✅ Performance Rating: {efficiency_metrics['performance_rating']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run comprehensive tests for all advanced features."""
    
    print("🚀 Advanced Solar Power Features - Comprehensive Test")
    print("=" * 70)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Test all features
    results = {}
    
    results['Feature 1'] = test_feature_1_ensemble_model()
    results['Feature 2'] = test_feature_2_roi_calculator()
    results['Feature 3'] = test_feature_3_3d_visualization()
    results['Feature 4'] = test_feature_4_satellite_integration()
    results['Feature 5'] = test_feature_5_carbon_tracker()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Results Summary")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for feature, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{feature:15} {status}")
        if result:
            passed += 1
    
    print("-" * 70)
    print(f"Overall: {passed}/{total} features passed")
    
    if passed == total:
        print("🎉 All advanced features are working correctly!")
        print("🚀 Your solar power prediction system is now enhanced with:")
        print("   • Multi-Model Ensemble with Dynamic Selection")
        print("   • ROI Calculator for Solar Installations")
        print("   • 3D Solar Irradiance Maps")
        print("   • Satellite Data Integration")
        print("   • Carbon Footprint Reduction Tracker")
    else:
        print("⚠️  Some features need attention. Check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)