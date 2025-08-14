import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_solar_power_data(n_samples=10000):
    """
    Generate synthetic solar power generation data with realistic weather features.
    This creates a comprehensive dataset for training the ML model.
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate dates (2 years of data)
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    # Generate realistic weather data
    data = []
    
    for i, date in enumerate(dates):
        # Seasonal patterns
        day_of_year = date.timetuple().tm_yday
        season_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Time of day effect (solar noon is around 12:00)
        hour = date.hour
        solar_noon_distance = abs(hour - 12)  # Distance from solar noon
        
        # Base temperature with seasonal and daily variations
        base_temp = 20 + 10 * season_factor
        temp = base_temp + 5 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 3)
        
        # Humidity (inverse relationship with temperature)
        humidity = max(20, min(95, 80 - 0.5 * temp + np.random.normal(0, 10)))
        
        # Wind speed (higher during day, lower at night)
        wind_speed = max(0, 2 + 3 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 2))
        
        # Wind direction (degrees, 0-360)
        wind_direction = np.random.uniform(0, 360)
        
        # Cloud cover (sky-cover) - affects solar generation significantly
        cloud_cover = np.random.uniform(0, 100)
        
        # Pressure (normal atmospheric pressure with some variation)
        pressure = 1013 + np.random.normal(0, 20)
        
        # Visibility (affected by weather conditions)
        visibility = max(0, 10 + np.random.normal(0, 3))
        
        # Solar power generation calculation
        # Base generation capacity (kW)
        base_capacity = 100
        
        # Solar irradiance factor (peak at solar noon)
        solar_factor = max(0, 1 - (solar_noon_distance / 6) ** 2)
        
        # Cloud cover effect (linear reduction)
        cloud_factor = 1 - (cloud_cover / 100) * 0.7
        
        # Temperature effect (solar panels are less efficient at high temps)
        temp_factor = 1 - 0.005 * max(0, temp - 25)
        
        # Humidity effect (slight reduction)
        humidity_factor = 1 - 0.001 * humidity
        
        # Calculate power generation
        power_generated = (base_capacity * 
                          solar_factor * 
                          cloud_factor * 
                          temp_factor * 
                          humidity_factor * 
                          season_factor)
        
        # Add some realistic noise
        power_generated += np.random.normal(0, power_generated * 0.1)
        power_generated = max(0, power_generated)
        
        data.append({
            'date': date,
            'temperature': round(temp, 2),
            'humidity': round(humidity, 2),
            'wind_speed': round(wind_speed, 2),
            'wind_direction': round(wind_direction, 2),
            'cloud_cover': round(cloud_cover, 2),
            'pressure': round(pressure, 2),
            'visibility': round(visibility, 2),
            'solar_noon_distance': round(solar_noon_distance, 2),
            'power_generated': round(power_generated, 2)
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate the dataset
    print("Generating solar power generation dataset...")
    df = generate_solar_power_data(10000)
    
    # Save to CSV
    df.to_csv('augmented_solarpowergeneration.csv', index=False)
    print(f"Dataset saved with {len(df)} samples")
    print(f"Features: {list(df.columns)}")
    print(f"Target variable: power_generated")
    print(f"Data range: {df['date'].min()} to {df['date'].max()}")
    print(f"Power generation range: {df['power_generated'].min():.2f} to {df['power_generated'].max():.2f} kW")