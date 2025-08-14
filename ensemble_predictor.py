#!/usr/bin/env python3
"""
Advanced Ensemble Predictor with Dynamic Model Selection
Uses meta-learner to select the optimal model for given weather conditions.
"""

import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

class AdvancedEnsemblePredictor:
    def __init__(self, model_dir='ensemble_models'):
        """Initialize the ensemble predictor."""
        self.model_dir = model_dir
        self.models = {}
        self.meta_learner = None
        self.scalers = {}
        self.feature_names = []
        self.metadata = {}
        self.model_names = []
        
        # Load the ensemble model
        self._load_ensemble_model()
    
    def _load_ensemble_model(self):
        """Load all ensemble model components."""
        try:
            # Load metadata
            self.metadata = joblib.load(f'{self.model_dir}/metadata.pkl')
            self.feature_names = self.metadata['feature_names']
            self.model_names = self.metadata['model_names']
            
            # Load meta-learner
            self.meta_learner = joblib.load(f'{self.model_dir}/meta_learner.pkl')
            
            # Load individual models
            for name in self.model_names:
                model_file = f'{self.model_dir}/{name.lower().replace(" ", "_")}.pkl'
                self.models[name] = joblib.load(model_file)
                
                # Load scaler if exists
                scaler_file = f'{self.model_dir}/{name.lower().replace(" ", "_")}_scaler.pkl'
                if os.path.exists(scaler_file):
                    self.scalers[name] = joblib.load(scaler_file)
            
            print(f"✅ Ensemble model loaded successfully")
            print(f"   Models: {len(self.models)}")
            print(f"   Features: {len(self.feature_names)}")
            print(f"   Training date: {self.metadata['training_date']}")
            
        except Exception as e:
            print(f"❌ Error loading ensemble model: {e}")
            raise
    
    def prepare_features(self, weather_data):
        """Prepare features from weather data for prediction."""
        try:
            # Extract basic features
            features = {
                'temperature': weather_data.get('temperature', 20),
                'humidity': weather_data.get('humidity', 50),
                'wind_speed': weather_data.get('wind_speed', 5),
                'wind_direction': weather_data.get('wind_direction', 180),
                'cloud_cover': weather_data.get('cloud_cover', 50),
                'pressure': weather_data.get('pressure', 1013),
                'visibility': weather_data.get('visibility', 10),
                'solar_noon_distance': weather_data.get('solar_noon_distance', 3)
            }
            
            # Calculate current time features
            current_time = datetime.now()
            hour = current_time.hour
            day_of_year = current_time.timetuple().tm_yday
            
            # Add time-based features
            features.update({
                'hour': hour,
                'day_of_year': day_of_year,
                'month': current_time.month,
                'season': current_time.month % 12 // 3 + 1
            })
            
            # Add cyclical features
            features.update({
                'hour_sin': np.sin(2 * np.pi * hour / 24),
                'hour_cos': np.cos(2 * np.pi * hour / 24),
                'day_sin': np.sin(2 * np.pi * day_of_year / 365),
                'day_cos': np.cos(2 * np.pi * day_of_year / 365),
                'wind_direction_sin': np.sin(2 * np.pi * features['wind_direction'] / 360),
                'wind_direction_cos': np.cos(2 * np.pi * features['wind_direction'] / 360)
            })
            
            # Add interaction features
            features.update({
                'temp_humidity_interaction': features['temperature'] * features['humidity'],
                'wind_pressure_interaction': features['wind_speed'] * features['pressure'],
                'cloud_visibility_interaction': features['cloud_cover'] * features['visibility'],
                'solar_efficiency': features['temperature'] * features['solar_noon_distance']
            })
            
            # Add weather condition (encoded as integer)
            if features['cloud_cover'] < 20:
                features['weather_condition'] = 0  # clear
            elif features['cloud_cover'] < 50:
                features['weather_condition'] = 1  # partly_cloudy
            elif features['cloud_cover'] < 80:
                features['weather_condition'] = 2  # cloudy
            else:
                features['weather_condition'] = 3  # overcast
            
            # Add solar potential
            if 6 <= hour <= 18:
                time_factor = 1 - (features['solar_noon_distance'] / 6) ** 2
            else:
                time_factor = 0
            
            weather_factor = 1 - (features['cloud_cover'] / 100) ** 1.5 * 0.8
            temp_factor = 1 - 0.004 * max(0, features['temperature'] - 25)
            
            features['solar_potential'] = time_factor * weather_factor * temp_factor
            
            # Create feature vector in correct order
            feature_vector = []
            for feature_name in self.feature_names:
                if feature_name in features:
                    feature_vector.append(features[feature_name])
                else:
                    feature_vector.append(0)  # Default value for missing features
            
            return np.array(feature_vector).reshape(1, -1)
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            raise
    
    def predict(self, weather_data):
        """Make prediction using the ensemble model with dynamic model selection."""
        try:
            # Prepare features
            features = self.prepare_features(weather_data)
            
            # Get predictions from all individual models
            model_predictions = {}
            for name, model in self.models.items():
                if name in self.scalers:
                    features_scaled = self.scalers[name].transform(features)
                    pred = model.predict(features_scaled)[0]
                else:
                    pred = model.predict(features)[0]
                model_predictions[name] = pred
            
            # Create meta-features for model selection
            meta_features = self._create_meta_features(features, model_predictions)
            
            # Get ensemble prediction from meta-learner
            ensemble_prediction = self.meta_learner.predict(meta_features)[0]
            
            # Apply realistic constraints
            ensemble_prediction = self._apply_constraints(ensemble_prediction, weather_data)
            
            # Calculate confidence and model selection info
            confidence, selected_model = self._calculate_confidence_and_selection(
                weather_data, model_predictions, ensemble_prediction
            )
            
            # Create result dictionary
            result = {
                'predicted_power': round(ensemble_prediction, 2),
                'confidence': round(confidence, 2),
                'selected_model': selected_model,
                'all_model_predictions': {name: round(pred, 2) for name, pred in model_predictions.items()},
                'model_used': 'Advanced Ensemble',
                'prediction_timestamp': datetime.now().isoformat(),
                'weather_conditions': {
                    'temperature': weather_data.get('temperature', 0),
                    'humidity': weather_data.get('humidity', 0),
                    'cloud_cover': weather_data.get('cloud_cover', 0),
                    'wind_speed': weather_data.get('wind_speed', 0),
                    'solar_noon_distance': weather_data.get('solar_noon_distance', 0)
                },
                'city': weather_data.get('city_name', 'Unknown'),
                'weather_description': weather_data.get('weather_description', 'Unknown'),
                'ensemble_info': {
                    'total_models': len(self.models),
                    'model_names': list(self.models.keys()),
                    'meta_learner_type': type(self.meta_learner).__name__
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            raise
    
    def _create_meta_features(self, features, model_predictions):
        """Create meta-features for the meta-learner."""
        # Weather condition features
        weather_features = self._extract_weather_features(features)
        
        # Model prediction features
        pred_features = np.array([model_predictions[name] for name in self.model_names])
        
        # Combine features
        meta_features = np.concatenate([weather_features.flatten(), pred_features])
        
        return meta_features.reshape(1, -1)
    
    def _extract_weather_features(self, features):
        """Extract weather condition features for meta-learning."""
        # Weather condition
        cloud_cover = features[0, self.feature_names.index('cloud_cover')]
        
        if cloud_cover < 20:
            weather_clear = 1
            weather_partly = 0
            weather_cloudy = 0
            weather_overcast = 0
        elif cloud_cover < 50:
            weather_clear = 0
            weather_partly = 1
            weather_cloudy = 0
            weather_overcast = 0
        elif cloud_cover < 80:
            weather_clear = 0
            weather_partly = 0
            weather_cloudy = 1
            weather_overcast = 0
        else:
            weather_clear = 0
            weather_partly = 0
            weather_cloudy = 0
            weather_overcast = 1
        
        # Time of day
        hour = features[0, self.feature_names.index('hour')]
        is_daytime = 1 if 6 <= hour <= 18 else 0
        
        # Solar potential
        solar_potential = features[0, self.feature_names.index('solar_potential')]
        
        # Other weather features
        temperature = features[0, self.feature_names.index('temperature')]
        humidity = features[0, self.feature_names.index('humidity')]
        wind_speed = features[0, self.feature_names.index('wind_speed')]
        
        return np.array([
            weather_clear, weather_partly, weather_cloudy, weather_overcast,
            is_daytime, solar_potential, temperature, humidity, wind_speed
        ])
    
    def _apply_constraints(self, prediction, weather_data):
        """Apply realistic constraints to the prediction."""
        solar_noon_distance = weather_data.get('solar_noon_distance', 0)
        cloud_cover = weather_data.get('cloud_cover', 50)
        
        # Check if it's daylight hours
        if solar_noon_distance > 6:
            prediction = 0  # No solar generation at night
        
        # Apply cloud cover effect
        if cloud_cover > 90:
            prediction *= 0.1  # Very cloudy - minimal generation
        elif cloud_cover > 70:
            prediction *= 0.3  # Cloudy - reduced generation
        elif cloud_cover > 50:
            prediction *= 0.6  # Partly cloudy - moderate reduction
        
        # Ensure prediction is non-negative and realistic
        prediction = max(0, min(100, prediction))  # Cap at 100 kW
        
        return prediction
    
    def _calculate_confidence_and_selection(self, weather_data, model_predictions, ensemble_prediction):
        """Calculate confidence and determine which model was most influential."""
        # Calculate variance in predictions
        predictions = list(model_predictions.values())
        variance = np.var(predictions)
        
        # Base confidence on weather conditions
        cloud_cover = weather_data.get('cloud_cover', 50)
        solar_noon_distance = weather_data.get('solar_noon_distance', 0)
        
        # Weather-based confidence
        if cloud_cover < 20:
            weather_confidence = 0.9  # Clear weather - high confidence
        elif cloud_cover < 50:
            weather_confidence = 0.8  # Partly cloudy - good confidence
        elif cloud_cover < 80:
            weather_confidence = 0.6  # Cloudy - moderate confidence
        else:
            weather_confidence = 0.4  # Overcast - low confidence
        
        # Time-based confidence
        if solar_noon_distance <= 2:
            time_confidence = 0.9  # Near solar noon - high confidence
        elif solar_noon_distance <= 4:
            time_confidence = 0.8  # Good daylight - good confidence
        elif solar_noon_distance <= 6:
            time_confidence = 0.6  # Early/late day - moderate confidence
        else:
            time_confidence = 0.0  # Night - no confidence
        
        # Variance-based confidence (lower variance = higher confidence)
        variance_confidence = max(0.1, 1 - variance / 100)
        
        # Overall confidence
        confidence = (weather_confidence + time_confidence + variance_confidence) / 3
        
        # Determine most influential model (closest to ensemble prediction)
        selected_model = min(model_predictions.keys(), 
                           key=lambda x: abs(model_predictions[x] - ensemble_prediction))
        
        return confidence, selected_model
    
    def get_model_info(self):
        """Get information about the ensemble model."""
        return {
            'model_name': 'Advanced Ensemble',
            'training_date': self.metadata['training_date'],
            'feature_count': len(self.feature_names),
            'model_count': len(self.models),
            'model_names': list(self.models.keys()),
            'meta_learner_type': type(self.meta_learner).__name__,
            'individual_performance': self.metadata['model_performance']
        }
    
    def get_feature_importance(self):
        """Get feature importance from the meta-learner."""
        if hasattr(self.meta_learner, 'feature_importances_'):
            return dict(zip(range(len(self.meta_learner.feature_importances_)), 
                           self.meta_learner.feature_importances_))
        return {}

if __name__ == "__main__":
    # Test the ensemble predictor
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
    print("Ensemble Prediction Result:")
    print(f"Predicted Power: {result['predicted_power']} kW")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Selected Model: {result['selected_model']}")
    print(f"All Model Predictions: {result['all_model_predictions']}")