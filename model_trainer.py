import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SolarPowerModelTrainer:
    def __init__(self, data_path='augmented_solarpowergeneration.csv'):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the solar power generation data."""
        print("Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(self.data_path)
        
        # Convert date to datetime features
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Create cyclical features for time
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Wind direction cyclical features
        df['wind_direction_sin'] = np.sin(2 * np.pi * df['wind_direction'] / 360)
        df['wind_direction_cos'] = np.cos(2 * np.pi * df['wind_direction'] / 360)
        
        # Feature engineering
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
        df['wind_pressure_interaction'] = df['wind_speed'] * df['pressure']
        df['cloud_visibility_interaction'] = df['cloud_cover'] * df['visibility']
        
        # Select features for modeling
        feature_columns = [
            'temperature', 'humidity', 'wind_speed', 'wind_direction',
            'cloud_cover', 'pressure', 'visibility', 'solar_noon_distance',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'wind_direction_sin', 'wind_direction_cos',
            'temp_humidity_interaction', 'wind_pressure_interaction',
            'cloud_visibility_interaction'
        ]
        
        X = df[feature_columns]
        y = df['power_generated']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = X_train
        self.X_test = X_test
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_columns
        
        print(f"Data shape: {X.shape}")
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def evaluate_models(self):
        """Evaluate multiple machine learning models."""
        print("\nEvaluating different models...")
        
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
            'LightGBM': LGBMRegressor(n_estimators=100, random_state=42),
            'CatBoost': CatBoostRegressor(iterations=100, random_state=42, verbose=False),
            'SVR': SVR(kernel='rbf'),
            'Linear Regression': LinearRegression()
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            if name in ['SVR', 'Linear Regression']:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Cross-validation score
            if name in ['SVR', 'Linear Regression']:
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                          cv=5, scoring='neg_mean_squared_error')
            else:
                cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                          cv=5, scoring='neg_mean_squared_error')
            
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            results[name] = {
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
                'RMSE': rmse,
                'CV_RMSE': cv_rmse,
                'model': model
            }
            
            print(f"  {name}: R²={r2:.4f}, RMSE={rmse:.4f}, CV_RMSE={cv_rmse:.4f}")
        
        return results
    
    def hyperparameter_tuning(self, model_name, model, param_grid):
        """Perform hyperparameter tuning for the best model."""
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        if model_name in ['SVR', 'Linear Regression']:
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(self.X_train_scaled, self.y_train)
        else:
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
        
        best_model = grid_search.best_estimator_
        best_score = np.sqrt(-grid_search.best_score_)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV RMSE: {best_score:.4f}")
        
        return best_model
    
    def select_best_model(self, results):
        """Select the best model based on R² and RMSE."""
        print("\nSelecting best model...")
        
        # Sort by R² score (higher is better)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['R2'], reverse=True)
        
        best_name = sorted_results[0][0]
        best_result = sorted_results[0][1]
        
        print(f"Best model: {best_name}")
        print(f"R² Score: {best_result['R2']:.4f}")
        print(f"RMSE: {best_result['RMSE']:.4f}")
        print(f"MAE: {best_result['MAE']:.4f}")
        
        self.best_model = best_result['model']
        self.best_model_name = best_name
        
        return best_name, best_result
    
    def get_feature_importance(self):
        """Get feature importance from the best model."""
        if self.best_model is None:
            print("No model trained yet!")
            return None
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importance = np.abs(self.best_model.coef_)
        else:
            print("Model doesn't support feature importance")
            return None
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = feature_importance
        return feature_importance
    
    def plot_results(self, results):
        """Plot model comparison and feature importance."""
        # Model comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # R² scores
        names = list(results.keys())
        r2_scores = [results[name]['R2'] for name in names]
        axes[0, 0].bar(names, r2_scores)
        axes[0, 0].set_title('R² Scores Comparison')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RMSE scores
        rmse_scores = [results[name]['RMSE'] for name in names]
        axes[0, 1].bar(names, rmse_scores)
        axes[0, 1].set_title('RMSE Scores Comparison')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Actual vs Predicted for best model
        if self.best_model is not None:
            if self.best_model_name in ['SVR', 'Linear Regression']:
                y_pred = self.best_model.predict(self.X_test_scaled)
            else:
                y_pred = self.best_model.predict(self.X_test)
            
            axes[1, 0].scatter(self.y_test, y_pred, alpha=0.6)
            axes[1, 0].plot([self.y_test.min(), self.y_test.max()], 
                           [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            axes[1, 0].set_xlabel('Actual Power Generated')
            axes[1, 0].set_ylabel('Predicted Power Generated')
            axes[1, 0].set_title(f'Actual vs Predicted ({self.best_model_name})')
        
        # Feature importance
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            axes[1, 1].barh(top_features['feature'], top_features['importance'])
            axes[1, 1].set_title('Top 10 Feature Importance')
            axes[1, 1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filename='solar_power_model.pkl'):
        """Save the best model and scaler."""
        if self.best_model is None:
            print("No model to save!")
            return
        
        # Save model and scaler
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': self.best_model_name,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved as {filename}")
        
        # Save feature importance
        if self.feature_importance is not None:
            self.feature_importance.to_csv('feature_importance.csv', index=False)
            print("Feature importance saved as feature_importance.csv")
    
    def train_and_evaluate(self):
        """Complete training and evaluation pipeline."""
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Evaluate models
        results = self.evaluate_models()
        
        # Select best model
        best_name, best_result = self.select_best_model(results)
        
        # Hyperparameter tuning for the best model
        if best_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        elif best_name == 'XGBoost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        elif best_name == 'LightGBM':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        else:
            param_grid = {}
        
        if param_grid:
            self.best_model = self.hyperparameter_tuning(best_name, best_result['model'], param_grid)
        
        # Get feature importance
        self.get_feature_importance()
        
        # Plot results
        self.plot_results(results)
        
        # Save model
        self.save_model()
        
        return self.best_model

if __name__ == "__main__":
    # Initialize trainer
    trainer = SolarPowerModelTrainer()
    
    # Train and evaluate models
    best_model = trainer.train_and_evaluate()
    
    print("\nTraining completed successfully!")
    print(f"Best model: {trainer.best_model_name}")
    print("Model saved as 'solar_power_model.pkl'")