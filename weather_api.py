import requests
import pandas as pd
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

class WeatherAPI:
    def __init__(self, api_key=None):
        """
        Initialize WeatherAPI with OpenWeatherMap API key.
        
        Args:
            api_key (str): OpenWeatherMap API key. If None, tries to get from environment.
        """
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key is required. Set OPENWEATHER_API_KEY environment variable or pass api_key parameter.")
        
        self.base_url = "http://api.openweathermap.org/data/2.5"
        
    def get_weather_data(self, city_name, country_code=None):
        """
        Fetch current weather data for a given city.
        
        Args:
            city_name (str): Name of the city
            country_code (str): Country code (optional, e.g., 'US', 'IN')
            
        Returns:
            dict: Weather data with processed features for solar power prediction
        """
        try:
            # Build location string
            location = f"{city_name},{country_code}" if country_code else city_name
            
            # Fetch current weather
            current_url = f"{self.base_url}/weather"
            params = {
                'q': location,
                'appid': self.api_key,
                'units': 'metric'  # Use metric units
            }
            
            response = requests.get(current_url, params=params, timeout=10)
            response.raise_for_status()
            
            weather_data = response.json()
            
            # Process and extract relevant features
            processed_data = self._process_weather_data(weather_data)
            
            return processed_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
        except Exception as e:
            print(f"Error processing weather data: {e}")
            return None
    
    def _process_weather_data(self, weather_data):
        """
        Process raw weather data and extract features needed for solar power prediction.
        
        Args:
            weather_data (dict): Raw weather data from OpenWeatherMap API
            
        Returns:
            dict: Processed weather features
        """
        try:
            # Extract basic weather information
            main = weather_data.get('main', {})
            wind = weather_data.get('wind', {})
            clouds = weather_data.get('clouds', {})
            weather = weather_data.get('weather', [{}])[0]
            sys = weather_data.get('sys', {})
            
            # Get current time
            current_time = datetime.now()
            hour = current_time.hour
            day_of_year = current_time.timetuple().tm_yday
            
            # Calculate solar noon distance
            solar_noon_distance = abs(hour - 12)
            
            # Create cyclical time features
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * day_of_year / 365)
            day_cos = np.cos(2 * np.pi * day_of_year / 365)
            
            # Process wind direction (convert to radians and create cyclical features)
            wind_direction = wind.get('deg', 0)
            wind_direction_sin = np.sin(2 * np.pi * wind_direction / 360)
            wind_direction_cos = np.cos(2 * np.pi * wind_direction / 360)
            
            # Extract and process features
            processed_data = {
                # Basic weather features
                'temperature': main.get('temp', 0),
                'humidity': main.get('humidity', 0),
                'pressure': main.get('pressure', 1013),
                'wind_speed': wind.get('speed', 0),
                'wind_direction': wind_direction,
                'cloud_cover': clouds.get('all', 0),  # This is sky-cover percentage
                'visibility': weather_data.get('visibility', 10000) / 1000,  # Convert to km
                'solar_noon_distance': solar_noon_distance,
                
                # Cyclical time features
                'hour_sin': hour_sin,
                'hour_cos': hour_cos,
                'day_sin': day_sin,
                'day_cos': day_cos,
                
                # Cyclical wind direction features
                'wind_direction_sin': wind_direction_sin,
                'wind_direction_cos': wind_direction_cos,
                
                # Interaction features
                'temp_humidity_interaction': main.get('temp', 0) * main.get('humidity', 0),
                'wind_pressure_interaction': wind.get('speed', 0) * main.get('pressure', 1013),
                'cloud_visibility_interaction': clouds.get('all', 0) * (weather_data.get('visibility', 10000) / 1000),
                
                # Additional metadata
                'city_name': weather_data.get('name', ''),
                'country': sys.get('country', ''),
                'weather_description': weather.get('description', ''),
                'weather_main': weather.get('main', ''),
                'timestamp': current_time.isoformat(),
                
                # Raw data for debugging
                'raw_data': weather_data
            }
            
            return processed_data
            
        except Exception as e:
            print(f"Error processing weather data: {e}")
            return None
    
    def get_forecast_data(self, city_name, country_code=None, days=5):
        """
        Fetch weather forecast data for a given city.
        
        Args:
            city_name (str): Name of the city
            country_code (str): Country code (optional)
            days (int): Number of days for forecast (max 5 for free API)
            
        Returns:
            list: List of forecast data points
        """
        try:
            # Build location string
            location = f"{city_name},{country_code}" if country_code else city_name
            
            # Fetch forecast
            forecast_url = f"{self.base_url}/forecast"
            params = {
                'q': location,
                'appid': self.api_key,
                'units': 'metric',
                'cnt': days * 8  # 8 data points per day (3-hour intervals)
            }
            
            response = requests.get(forecast_url, params=params, timeout=10)
            response.raise_for_status()
            
            forecast_data = response.json()
            
            # Process forecast data
            processed_forecast = []
            for item in forecast_data.get('list', []):
                processed_item = self._process_forecast_item(item)
                if processed_item:
                    processed_forecast.append(processed_item)
            
            return processed_forecast
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching forecast data: {e}")
            return None
        except Exception as e:
            print(f"Error processing forecast data: {e}")
            return None
    
    def _process_forecast_item(self, forecast_item):
        """
        Process individual forecast data point.
        
        Args:
            forecast_item (dict): Single forecast data point
            
        Returns:
            dict: Processed forecast data
        """
        try:
            # Extract timestamp
            timestamp = datetime.fromtimestamp(forecast_item.get('dt', 0))
            hour = timestamp.hour
            day_of_year = timestamp.timetuple().tm_yday
            
            # Calculate solar noon distance
            solar_noon_distance = abs(hour - 12)
            
            # Create cyclical features
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * day_of_year / 365)
            day_cos = np.cos(2 * np.pi * day_of_year / 365)
            
            # Extract weather data
            main = forecast_item.get('main', {})
            wind = forecast_item.get('wind', {})
            clouds = forecast_item.get('clouds', {})
            
            # Process wind direction
            wind_direction = wind.get('deg', 0)
            wind_direction_sin = np.sin(2 * np.pi * wind_direction / 360)
            wind_direction_cos = np.cos(2 * np.pi * wind_direction / 360)
            
            processed_item = {
                'timestamp': timestamp.isoformat(),
                'temperature': main.get('temp', 0),
                'humidity': main.get('humidity', 0),
                'pressure': main.get('pressure', 1013),
                'wind_speed': wind.get('speed', 0),
                'wind_direction': wind_direction,
                'cloud_cover': clouds.get('all', 0),
                'solar_noon_distance': solar_noon_distance,
                'hour_sin': hour_sin,
                'hour_cos': hour_cos,
                'day_sin': day_sin,
                'day_cos': day_cos,
                'wind_direction_sin': wind_direction_sin,
                'wind_direction_cos': wind_direction_cos,
                'temp_humidity_interaction': main.get('temp', 0) * main.get('humidity', 0),
                'wind_pressure_interaction': wind.get('speed', 0) * main.get('pressure', 1013),
                'cloud_visibility_interaction': clouds.get('all', 0) * 10,  # Default visibility
            }
            
            return processed_item
            
        except Exception as e:
            print(f"Error processing forecast item: {e}")
            return None
    
    def validate_api_key(self):
        """
        Validate the API key by making a test request.
        
        Returns:
            bool: True if API key is valid, False otherwise
        """
        try:
            # Make a test request to a known city
            test_data = self.get_weather_data('London')
            return test_data is not None
        except Exception:
            return False

# Example usage and testing
if __name__ == "__main__":
    # Test the API (you'll need to set your API key)
    try:
        weather_api = WeatherAPI()
        
        # Test current weather
        print("Testing current weather for London...")
        current_weather = weather_api.get_weather_data('London')
        if current_weather:
            print("Current weather data:")
            for key, value in current_weather.items():
                if key != 'raw_data':
                    print(f"  {key}: {value}")
        
        # Test forecast
        print("\nTesting forecast for London...")
        forecast = weather_api.get_forecast_data('London', days=1)
        if forecast:
            print(f"Forecast data points: {len(forecast)}")
            print("Sample forecast point:")
            for key, value in forecast[0].items():
                print(f"  {key}: {value}")
                
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your OpenWeatherMap API key in the environment variable OPENWEATHER_API_KEY")