#!/usr/bin/env python3
"""
Advanced Ensemble Model Trainer with Dynamic Model Selection
Implements multiple ML models and a meta-learner for optimal model selection.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class AdvancedEnsembleTrainer:
    def __init__(self):
        self.models = {}
        self.meta_learner = None
        self.scalers = {}
        self.feature_names = []
        self.model_performance = {}
        self.weather_conditions = {}
        
    def load_and_preprocess_data(self, csv_file='augmented_solarpowergeneration.csv'):
        """Load and preprocess the dataset with advanced feature engineering."""
        print("📊 Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(csv_file)
        print(f"Original data shape: {df.shape}")
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Advanced feature engineering
        df = self._create_advanced_features(df)
        
        # Prepare features and target
        feature_columns = [col for col in df.columns if col not in ['date', 'power_generated']]
        self.feature_names = feature_columns
        
        X = df[feature_columns]
        y = df['power_generated']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=10, labels=False)
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, df
    
    def _create_advanced_features(self, df):
        """Create advanced features for better model performance."""
        
        # Time-based features
        df['hour'] = df['date'].dt.hour
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['season'] = df['date'].dt.month % 12 // 3 + 1
        
        # Cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        df['wind_direction_sin'] = np.sin(2 * np.pi * df['wind_direction'] / 360)
        df['wind_direction_cos'] = np.cos(2 * np.pi * df['wind_direction'] / 360)
        
        # Advanced interaction features
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
        df['wind_pressure_interaction'] = df['wind_speed'] * df['pressure']
        df['cloud_visibility_interaction'] = df['cloud_cover'] * df['visibility']
        df['solar_efficiency'] = df['temperature'] * df['solar_noon_distance']
        
        # Weather condition categories (encoded as integers)
        df['weather_condition'] = self._categorize_weather(df)
        
        # Solar potential features
        df['solar_potential'] = self._calculate_solar_potential(df)
        
        # Drop original date column
        df = df.drop('date', axis=1)
        
        return df
    
    def _categorize_weather(self, df):
        """Categorize weather conditions for model selection."""
        conditions = []
        for _, row in df.iterrows():
            if row['cloud_cover'] < 20:
                conditions.append(0)  # clear
            elif row['cloud_cover'] < 50:
                conditions.append(1)  # partly_cloudy
            elif row['cloud_cover'] < 80:
                conditions.append(2)  # cloudy
            else:
                conditions.append(3)  # overcast
        return conditions
    
    def _calculate_solar_potential(self, df):
        """Calculate solar potential based on time and weather."""
        potential = []
        for _, row in df.iterrows():
            # Base potential from time of day
            if 6 <= row['hour'] <= 18:
                time_factor = 1 - (row['solar_noon_distance'] / 6) ** 2
            else:
                time_factor = 0
            
            # Weather factor
            weather_factor = 1 - (row['cloud_cover'] / 100) ** 1.5 * 0.8
            
            # Temperature factor
            temp_factor = 1 - 0.004 * max(0, row['temperature'] - 25)
            
            potential.append(time_factor * weather_factor * temp_factor)
        
        return potential
    
    def train_individual_models(self, X_train, y_train):
        """Train multiple individual models."""
        print("\n🤖 Training individual models...")
        
        # Define models
        model_configs = {
            'Random Forest': {
                'model': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                'needs_scaling': False
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(n_estimators=200, random_state=42),
                'needs_scaling': False
            },
            'XGBoost': {
                'model': xgb.XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                'needs_scaling': False
            },
            'SVR': {
                'model': SVR(kernel='rbf', C=100, gamma='scale'),
                'needs_scaling': True
            },
            'Linear Regression': {
                'model': LinearRegression(),
                'needs_scaling': True
            }
        }
        
        # Train each model
        for name, config in model_configs.items():
            print(f"  Training {name}...")
            
            model = config['model']
            
            if config['needs_scaling']:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                self.scalers[name] = scaler
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train, y_train)
            
            self.models[name] = model
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            self.model_performance[name] = {
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std()
            }
            
            print(f"    {name}: CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    def train_meta_learner(self, X_train, y_train, X_test, y_test):
        """Train a meta-learner to select the best model for given conditions."""
        print("\n🧠 Training meta-learner for dynamic model selection...")
        
        # Generate predictions from all models
        model_predictions = {}
        
        for name, model in self.models.items():
            if name in self.scalers:
                X_train_scaled = self.scalers[name].transform(X_train)
                X_test_scaled = self.scalers[name].transform(X_test)
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
            else:
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
            
            model_predictions[name] = {
                'train': train_pred,
                'test': test_pred
            }
        
        # Create meta-features for model selection
        meta_features = self._create_meta_features(X_train, X_test, model_predictions)
        
        # Train meta-learner (Random Forest for model selection)
        self.meta_learner = RandomForestRegressor(n_estimators=100, random_state=42)
        self.meta_learner.fit(meta_features['train'], y_train)
        
        print("✅ Meta-learner trained successfully")
    
    def _create_meta_features(self, X_train, X_test, model_predictions):
        """Create meta-features for the meta-learner."""
        
        # Weather condition features
        weather_features_train = self._extract_weather_features(X_train)
        weather_features_test = self._extract_weather_features(X_test)
        
        # Model prediction features
        pred_features_train = np.column_stack([
            model_predictions[name]['train'] for name in self.models.keys()
        ])
        pred_features_test = np.column_stack([
            model_predictions[name]['test'] for name in self.models.keys()
        ])
        
        # Combine features
        meta_train = np.column_stack([weather_features_train, pred_features_train])
        meta_test = np.column_stack([weather_features_test, pred_features_test])
        
        return {
            'train': meta_train,
            'test': meta_test
        }
    
    def _extract_weather_features(self, X):
        """Extract weather condition features for meta-learning."""
        features = []
        
        for _, row in X.iterrows():
            # Weather condition
            if row['cloud_cover'] < 20:
                weather_clear = 1
                weather_partly = 0
                weather_cloudy = 0
                weather_overcast = 0
            elif row['cloud_cover'] < 50:
                weather_clear = 0
                weather_partly = 1
                weather_cloudy = 0
                weather_overcast = 0
            elif row['cloud_cover'] < 80:
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
            is_daytime = 1 if 6 <= row['hour'] <= 18 else 0
            
            # Solar potential
            solar_potential = row['solar_potential']
            
            features.append([
                weather_clear, weather_partly, weather_cloudy, weather_overcast,
                is_daytime, solar_potential,
                row['temperature'], row['humidity'], row['wind_speed']
            ])
        
        return np.array(features)
    
    def evaluate_ensemble(self, X_test, y_test):
        """Evaluate the ensemble model performance."""
        print("\n📊 Evaluating ensemble model...")
        
        # Get meta-features for test set
        model_predictions = {}
        for name, model in self.models.items():
            if name in self.scalers:
                X_test_scaled = self.scalers[name].transform(X_test)
                test_pred = model.predict(X_test_scaled)
            else:
                test_pred = model.predict(X_test)
            model_predictions[name] = {'test': test_pred}
        
        # Create meta-features for test set
        weather_features_test = self._extract_weather_features(X_test)
        pred_features_test = np.column_stack([
            model_predictions[name]['test'] for name in self.models.keys()
        ])
        meta_features_test = np.column_stack([weather_features_test, pred_features_test])
        
        # Get ensemble prediction
        ensemble_pred = self.meta_learner.predict(meta_features_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, ensemble_pred)
        mae = mean_absolute_error(y_test, ensemble_pred)
        r2 = r2_score(y_test, ensemble_pred)
        rmse = np.sqrt(mse)
        
        print(f"Ensemble Model Performance:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'predictions': ensemble_pred
        }
    
    def save_ensemble_model(self):
        """Save the ensemble model and metadata."""
        print("\n💾 Saving ensemble model...")
        
        # Create model directory
        os.makedirs('ensemble_models', exist_ok=True)
        
        # Save individual models
        for name, model in self.models.items():
            joblib.dump(model, f'ensemble_models/{name.lower().replace(" ", "_")}.pkl')
        
        # Save meta-learner
        joblib.dump(self.meta_learner, 'ensemble_models/meta_learner.pkl')
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f'ensemble_models/{name.lower().replace(" ", "_")}_scaler.pkl')
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'model_performance': self.model_performance,
            'training_date': datetime.now().isoformat(),
            'model_names': list(self.models.keys())
        }
        joblib.dump(metadata, 'ensemble_models/metadata.pkl')
        
        print("✅ Ensemble model saved successfully")
    
    def train_ensemble(self):
        """Complete ensemble training pipeline."""
        print("=" * 60)
        print("🚀 Advanced Ensemble Model Training")
        print("=" * 60)
        
        # Load and preprocess data
        X_train, X_test, y_train, y_test, df = self.load_and_preprocess_data()
        
        # Train individual models
        self.train_individual_models(X_train, y_train)
        
        # Train meta-learner
        self.train_meta_learner(X_train, y_train, X_test, y_test)
        
        # Evaluate ensemble
        results = self.evaluate_ensemble(X_test, y_test)
        
        # Save model
        self.save_ensemble_model()
        
        print("\n" + "=" * 60)
        print("🎉 Ensemble Training Completed Successfully!")
        print("=" * 60)
        print(f"Best Ensemble Performance:")
        print(f"  R² Score: {results['r2']:.4f}")
        print(f"  RMSE: {results['rmse']:.4f}")
        print(f"  MAE: {results['mae']:.4f}")
        
        return results

if __name__ == "__main__":
    trainer = AdvancedEnsembleTrainer()
    trainer.train_ensemble()