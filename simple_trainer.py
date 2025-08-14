#!/usr/bin/env python3
"""
Simplified Solar Power Generation Model Trainer
This script trains a focused set of reliable models for solar power prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(data_path='augmented_solarpowergeneration.csv'):
    """Load and preprocess the solar power generation data."""
    print("Loading and preprocessing data...")
    
    # Load data
    df = pd.read_csv(data_path)
    
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
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Data shape: {X.shape}")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns

def evaluate_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
    """Evaluate multiple machine learning models."""
    print("\nEvaluating different models...")
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf'),
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        try:
            # Train model
            if name in ['SVR', 'Linear Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Cross-validation score
            if name in ['SVR', 'Linear Regression']:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                          cv=5, scoring='neg_mean_squared_error')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, 
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
            
        except Exception as e:
            print(f"  {name}: Error - {e}")
            continue
    
    return results

def select_best_model(results):
    """Select the best model based on R² and RMSE."""
    print("\nSelecting best model...")
    
    if not results:
        print("No models were successfully trained!")
        return None, None
    
    # Sort by R² score (higher is better)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['R2'], reverse=True)
    
    best_name = sorted_results[0][0]
    best_result = sorted_results[0][1]
    
    print(f"Best model: {best_name}")
    print(f"R² Score: {best_result['R2']:.4f}")
    print(f"RMSE: {best_result['RMSE']:.4f}")
    print(f"MAE: {best_result['MAE']:.4f}")
    
    return best_name, best_result

def get_feature_importance(model, feature_names):
    """Get feature importance from the model."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        print("Model doesn't support feature importance")
        return None
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return feature_importance

def plot_results(results, best_model, X_test, y_test, X_test_scaled, best_model_name):
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
    if best_model is not None:
        if best_model_name in ['SVR', 'Linear Regression']:
            y_pred = best_model.predict(X_test_scaled)
        else:
            y_pred = best_model.predict(X_test)
        
        axes[1, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[1, 0].plot([y_test.min(), y_test.max()], 
                       [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Power Generated')
        axes[1, 0].set_ylabel('Predicted Power Generated')
        axes[1, 0].set_title(f'Actual vs Predicted ({best_model_name})')
    
    # Feature importance
    if best_model is not None:
        feature_importance = get_feature_importance(best_model, [
            'temperature', 'humidity', 'wind_speed', 'wind_direction',
            'cloud_cover', 'pressure', 'visibility', 'solar_noon_distance',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'wind_direction_sin', 'wind_direction_cos',
            'temp_humidity_interaction', 'wind_pressure_interaction',
            'cloud_visibility_interaction'
        ])
        
        if feature_importance is not None:
            top_features = feature_importance.head(10)
            axes[1, 1].barh(top_features['feature'], top_features['importance'])
            axes[1, 1].set_title('Top 10 Feature Importance')
            axes[1, 1].set_xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_model(best_model, scaler, feature_names, best_model_name, filename='solar_power_model.pkl'):
    """Save the best model and scaler."""
    if best_model is None:
        print("No model to save!")
        return
    
    # Save model and scaler
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'model_name': best_model_name,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    joblib.dump(model_data, filename)
    print(f"Model saved as {filename}")
    
    # Save feature importance
    feature_importance = get_feature_importance(best_model, feature_names)
    if feature_importance is not None:
        feature_importance.to_csv('feature_importance.csv', index=False)
        print("Feature importance saved as feature_importance.csv")

def main():
    """Main training function."""
    print("=" * 60)
    print("☀️ Simplified Solar Power Generation Model Training")
    print("=" * 60)
    
    # Load and preprocess data
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = load_and_preprocess_data()
    
    # Evaluate models
    results = evaluate_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Select best model
    best_model_name, best_result = select_best_model(results)
    
    if best_result is None:
        print("No models were successfully trained!")
        return
    
    best_model = best_result['model']
    
    # Plot results
    plot_results(results, best_model, X_test, y_test, X_test_scaled, best_model_name)
    
    # Save model
    save_model(best_model, scaler, feature_names, best_model_name)
    
    print("\n" + "=" * 60)
    print("🎉 Training completed successfully!")
    print("=" * 60)
    print(f"Best model: {best_model_name}")
    print("Model saved as 'solar_power_model.pkl'")
    print("Feature importance saved as 'feature_importance.csv'")
    print("Evaluation plots saved as 'model_evaluation_results.png'")

if __name__ == "__main__":
    main()