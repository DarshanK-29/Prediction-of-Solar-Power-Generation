import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SolarPowerPredictor:
    def __init__(self, model_path='solar_power_model.pkl'):
        """
        Initialize the solar power predictor with a trained model.
        
        Args:
            model_path (str): Path to the saved model file
        """
        self.model_path = model_path
        self.model_data = None
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_name = None
        
        # Load the model
        self.load_model()
    
    def load_model(self):
        """Load the trained model and associated data."""
        try:
            self.model_data = joblib.load(self.model_path)
            self.model = self.model_data['model']
            self.scaler = self.model_data['scaler']
            self.feature_names = self.model_data['feature_names']
            self.model_name = self.model_data['model_name']
            
            print(f"Model loaded successfully: {self.model_name}")
            print(f"Features: {len(self.feature_names)}")
            print(f"Training date: {self.model_data.get('training_date', 'Unknown')}")
            
        except FileNotFoundError:
            print(f"Model file not found: {self.model_path}")
            print("Please train the model first using model_trainer.py")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def prepare_features(self, weather_data):
        """
        Prepare weather data features for prediction.
        
        Args:
            weather_data (dict): Weather data from WeatherAPI
            
        Returns:
            numpy.ndarray: Prepared features for prediction
        """
        try:
            # Extract features in the same order as training
            features = []
            for feature_name in self.feature_names:
                if feature_name in weather_data:
                    features.append(weather_data[feature_name])
                else:
                    print(f"Warning: Feature '{feature_name}' not found in weather data")
                    features.append(0)  # Default value
            
            # Convert to numpy array and reshape
            features_array = np.array(features).reshape(1, -1)
            
            return features_array
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            raise
    
    def predict(self, weather_data):
        """
        Make solar power generation prediction.
        
        Args:
            weather_data (dict): Weather data from WeatherAPI
            
        Returns:
            dict: Prediction results with confidence metrics
        """
        try:
            # Prepare features
            features = self.prepare_features(weather_data)
            
            # Scale features if needed
            if self.model_name in ['SVR', 'Linear Regression']:
                features_scaled = self.scaler.transform(features)
                prediction = self.model.predict(features_scaled)[0]
            else:
                prediction = self.model.predict(features)[0]
            
            # Ensure prediction is non-negative
            prediction = max(0, prediction)
            
            # Calculate confidence based on weather conditions
            confidence = self._calculate_confidence(weather_data)
            
            # Create result dictionary
            result = {
                'predicted_power': round(prediction, 2),
                'confidence': round(confidence, 2),
                'model_used': self.model_name,
                'prediction_timestamp': datetime.now().isoformat(),
                'weather_conditions': {
                    'temperature': weather_data.get('temperature', 0),
                    'humidity': weather_data.get('humidity', 0),
                    'cloud_cover': weather_data.get('cloud_cover', 0),
                    'wind_speed': weather_data.get('wind_speed', 0),
                    'solar_noon_distance': weather_data.get('solar_noon_distance', 0)
                },
                'city': weather_data.get('city_name', 'Unknown'),
                'weather_description': weather_data.get('weather_description', 'Unknown')
            }
            
            return result
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            raise
    
    def _calculate_confidence(self, weather_data):
        """
        Calculate prediction confidence based on weather conditions.
        
        Args:
            weather_data (dict): Weather data
            
        Returns:
            float: Confidence score (0-1)
        """
        try:
            # Base confidence
            confidence = 0.8
            
            # Reduce confidence for extreme weather conditions
            temp = weather_data.get('temperature', 20)
            humidity = weather_data.get('humidity', 50)
            cloud_cover = weather_data.get('cloud_cover', 50)
            wind_speed = weather_data.get('wind_speed', 5)
            solar_distance = weather_data.get('solar_noon_distance', 6)
            
            # Temperature effect (optimal around 25°C)
            temp_factor = 1 - 0.01 * abs(temp - 25)
            confidence *= temp_factor
            
            # Cloud cover effect (more clouds = less confidence)
            cloud_factor = 1 - 0.002 * cloud_cover
            confidence *= cloud_factor
            
            # Wind speed effect (high winds can affect panels)
            wind_factor = 1 - 0.01 * max(0, wind_speed - 10)
            confidence *= wind_factor
            
            # Time of day effect (less confidence far from solar noon)
            time_factor = 1 - 0.02 * solar_distance
            confidence *= time_factor
            
            # Ensure confidence is between 0 and 1
            confidence = max(0.1, min(1.0, confidence))
            
            return confidence
            
        except Exception as e:
            print(f"Error calculating confidence: {e}")
            return 0.5  # Default confidence
    
    def predict_batch(self, weather_data_list):
        """
        Make predictions for multiple weather data points.
        
        Args:
            weather_data_list (list): List of weather data dictionaries
            
        Returns:
            list: List of prediction results
        """
        try:
            predictions = []
            
            for weather_data in weather_data_list:
                prediction = self.predict(weather_data)
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            print(f"Error making batch predictions: {e}")
            raise
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        return {
            'model_name': self.model_name,
            'feature_count': len(self.feature_names),
            'features': self.feature_names,
            'training_date': self.model_data.get('training_date', 'Unknown'),
            'model_path': self.model_path
        }
    
    def validate_prediction_input(self, weather_data):
        """
        Validate that weather data contains all required features.
        
        Args:
            weather_data (dict): Weather data to validate
            
        Returns:
            tuple: (is_valid, missing_features)
        """
        missing_features = []
        
        for feature_name in self.feature_names:
            if feature_name not in weather_data:
                missing_features.append(feature_name)
        
        is_valid = len(missing_features) == 0
        
        return is_valid, missing_features

# Example usage
if __name__ == "__main__":
    # Test the predictor
    try:
        # Initialize predictor
        predictor = SolarPowerPredictor()
        
        # Sample weather data (you would get this from WeatherAPI)
        sample_weather = {
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
            'cloud_visibility_interaction': 300.0,
            'city_name': 'Test City',
            'weather_description': 'Partly cloudy'
        }
        
        # Make prediction
        result = predictor.predict(sample_weather)
        
        print("Prediction Result:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
        # Get model info
        model_info = predictor.get_model_info()
        print("\nModel Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error testing predictor: {e}")
        print("Make sure you have trained the model first using model_trainer.py")