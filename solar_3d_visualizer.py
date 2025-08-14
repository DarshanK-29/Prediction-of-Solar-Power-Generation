#!/usr/bin/env python3
"""
3D Solar Irradiance Visualization System
Creates interactive 3D visualizations of solar potential across different times and conditions.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import math

class Solar3DVisualizer:
    def __init__(self):
        """Initialize the 3D solar visualizer."""
        self.colors = {
            'low': '#ff6b6b',      # Red for low irradiance
            'medium': '#feca57',   # Orange for medium irradiance
            'high': '#48dbfb',     # Blue for high irradiance
            'peak': '#0abde3'      # Dark blue for peak irradiance
        }
    
    def create_3d_irradiance_map(self, latitude, longitude, date=None, weather_conditions=None):
        """
        Create a 3D solar irradiance map for a specific location and time.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            date: Date for the visualization (default: today)
            weather_conditions: Weather conditions affecting irradiance
            
        Returns:
            plotly.graph_objects.Figure: 3D visualization
        """
        
        if date is None:
            date = datetime.now()
        
        if weather_conditions is None:
            weather_conditions = {
                'cloud_cover': 20,
                'temperature': 25,
                'humidity': 50,
                'visibility': 10
            }
        
        # Generate time grid (24 hours x 365 days)
        hours = np.arange(0, 24, 0.5)
        days = np.arange(1, 366, 1)
        
        # Create meshgrid
        H, D = np.meshgrid(hours, days)
        
        # Calculate solar irradiance for each point
        irradiance = np.zeros_like(H)
        
        for i, day in enumerate(days):
            for j, hour in enumerate(hours):
                irradiance[i, j] = self._calculate_solar_irradiance(
                    latitude, longitude, day, hour, weather_conditions
                )
        
        # Create 3D surface plot
        fig = go.Figure()
        
        # Add surface plot
        fig.add_trace(go.Surface(
            x=H,
            y=D,
            z=irradiance,
            colorscale='Viridis',
            name='Solar Irradiance',
            showscale=True,
            colorbar=dict(
                title="Irradiance (W/m²)",
                titleside="right",
                thickness=15,
                len=0.5
            )
        ))
        
        # Update layout
        fig.update_layout(
            title=f'3D Solar Irradiance Map - {date.strftime("%B %d, %Y")}',
            scene=dict(
                xaxis_title='Hour of Day',
                yaxis_title='Day of Year',
                zaxis_title='Solar Irradiance (W/m²)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def _calculate_solar_irradiance(self, lat, lon, day_of_year, hour, weather_conditions):
        """
        Calculate solar irradiance for given parameters.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            day_of_year: Day of year (1-365)
            hour: Hour of day (0-23)
            weather_conditions: Weather conditions dict
            
        Returns:
            float: Solar irradiance in W/m²
        """
        
        # Convert to radians
        lat_rad = math.radians(lat)
        
        # Calculate solar declination
        declination = 23.45 * math.sin(math.radians(360/365 * (day_of_year - 80)))
        declination_rad = math.radians(declination)
        
        # Calculate hour angle
        solar_noon = 12  # Solar noon is at 12:00
        hour_angle = 15 * (hour - solar_noon)  # 15 degrees per hour
        hour_angle_rad = math.radians(hour_angle)
        
        # Calculate solar altitude
        sin_altitude = (math.sin(lat_rad) * math.sin(declination_rad) + 
                       math.cos(lat_rad) * math.cos(declination_rad) * math.cos(hour_angle_rad))
        altitude = math.asin(max(-1, min(1, sin_altitude)))
        
        # Calculate air mass
        if altitude > 0:
            air_mass = 1 / math.sin(altitude)
        else:
            air_mass = float('inf')
        
        # Calculate extraterrestrial irradiance
        solar_constant = 1367  # W/m²
        earth_sun_distance = 1 + 0.034 * math.cos(math.radians(360/365 * (day_of_year - 2)))
        extraterrestrial_irradiance = solar_constant * earth_sun_distance
        
        # Calculate direct normal irradiance (simplified)
        if altitude > 0 and air_mass < 10:
            # Simplified atmospheric transmission
            transmission = 0.7 ** air_mass
            direct_irradiance = extraterrestrial_irradiance * transmission * math.sin(altitude)
        else:
            direct_irradiance = 0
        
        # Apply weather effects
        cloud_cover = weather_conditions.get('cloud_cover', 0)
        cloud_factor = 1 - (cloud_cover / 100) ** 1.5 * 0.8
        
        # Temperature effect
        temperature = weather_conditions.get('temperature', 25)
        temp_factor = 1 - 0.004 * max(0, temperature - 25)
        
        # Humidity effect
        humidity = weather_conditions.get('humidity', 50)
        humidity_factor = 1 - 0.0005 * humidity
        
        # Calculate final irradiance
        irradiance = direct_irradiance * cloud_factor * temp_factor * humidity_factor
        
        return max(0, irradiance)
    
    def create_daily_irradiance_profile(self, latitude, longitude, date=None, weather_conditions=None):
        """
        Create a daily solar irradiance profile.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            date: Date for the profile
            weather_conditions: Weather conditions
            
        Returns:
            plotly.graph_objects.Figure: Daily profile visualization
        """
        
        if date is None:
            date = datetime.now()
        
        if weather_conditions is None:
            weather_conditions = {
                'cloud_cover': 20,
                'temperature': 25,
                'humidity': 50,
                'visibility': 10
            }
        
        # Generate hourly data
        hours = np.arange(0, 24, 0.1)
        day_of_year = date.timetuple().tm_yday
        
        irradiance_values = []
        for hour in hours:
            irradiance = self._calculate_solar_irradiance(
                latitude, longitude, day_of_year, hour, weather_conditions
            )
            irradiance_values.append(irradiance)
        
        # Create figure
        fig = go.Figure()
        
        # Add irradiance line
        fig.add_trace(go.Scatter(
            x=hours,
            y=irradiance_values,
            mode='lines',
            name='Solar Irradiance',
            line=dict(color='#ff6b35', width=3),
            fill='tonexty',
            fillcolor='rgba(255, 107, 53, 0.3)'
        ))
        
        # Add peak sun hours
        peak_threshold = max(irradiance_values) * 0.8
        peak_hours = [h for h, i in zip(hours, irradiance_values) if i >= peak_threshold]
        
        if peak_hours:
            fig.add_trace(go.Scatter(
                x=peak_hours,
                y=[peak_threshold] * len(peak_hours),
                mode='markers',
                name='Peak Sun Hours',
                marker=dict(color='#ffd93d', size=8, symbol='diamond')
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Daily Solar Irradiance Profile - {date.strftime("%B %d, %Y")}',
            xaxis_title='Hour of Day',
            yaxis_title='Solar Irradiance (W/m²)',
            xaxis=dict(range=[0, 24]),
            yaxis=dict(range=[0, max(irradiance_values) * 1.1]),
            showlegend=True,
            width=800,
            height=400
        )
        
        return fig
    
    def create_seasonal_comparison(self, latitude, longitude, weather_conditions=None):
        """
        Create a seasonal comparison of solar irradiance.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            weather_conditions: Weather conditions
            
        Returns:
            plotly.graph_objects.Figure: Seasonal comparison visualization
        """
        
        if weather_conditions is None:
            weather_conditions = {
                'cloud_cover': 20,
                'temperature': 25,
                'humidity': 50,
                'visibility': 10
            }
        
        # Define seasons
        seasons = {
            'Spring': (80, 172),   # March 21 - June 21
            'Summer': (172, 266),  # June 21 - September 23
            'Autumn': (266, 355),  # September 23 - December 21
            'Winter': (355, 80)    # December 21 - March 21
        }
        
        # Generate data for each season
        hours = np.arange(6, 18, 0.5)  # Daylight hours only
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(seasons.keys()),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['#ff6b35', '#feca57', '#48dbfb', '#0abde3']
        
        for i, (season, (start_day, end_day)) in enumerate(seasons.items()):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            # Calculate average irradiance for the season
            season_irradiance = []
            for hour in hours:
                daily_irradiance = []
                for day in range(start_day, end_day + 1):
                    if day <= 365:
                        irradiance = self._calculate_solar_irradiance(
                            latitude, longitude, day, hour, weather_conditions
                        )
                        daily_irradiance.append(irradiance)
                
                if daily_irradiance:
                    season_irradiance.append(np.mean(daily_irradiance))
                else:
                    season_irradiance.append(0)
            
            fig.add_trace(
                go.Scatter(
                    x=hours,
                    y=season_irradiance,
                    mode='lines',
                    name=season,
                    line=dict(color=colors[i], width=3),
                    fill='tonexty',
                    fillcolor=f'rgba{tuple(int(colors[i][1:][j:j+2], 16) for j in (0, 2, 4)) + (0.3,)}'
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=f'Seasonal Solar Irradiance Comparison - Lat: {latitude}°, Lon: {longitude}°',
            showlegend=False,
            width=1000,
            height=600
        )
        
        # Update axes labels
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(title_text="Hour of Day", row=i, col=j)
                fig.update_yaxes(title_text="Irradiance (W/m²)", row=i, col=j)
        
        return fig
    
    def create_weather_impact_analysis(self, latitude, longitude, date=None):
        """
        Create weather impact analysis on solar irradiance.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            date: Date for analysis
            
        Returns:
            plotly.graph_objects.Figure: Weather impact visualization
        """
        
        if date is None:
            date = datetime.now()
        
        day_of_year = date.timetuple().tm_yday
        hours = np.arange(6, 18, 0.5)
        
        # Define weather scenarios
        weather_scenarios = {
            'Clear Sky': {'cloud_cover': 10, 'temperature': 25, 'humidity': 40},
            'Partly Cloudy': {'cloud_cover': 40, 'temperature': 22, 'humidity': 60},
            'Cloudy': {'cloud_cover': 70, 'temperature': 20, 'humidity': 75},
            'Overcast': {'cloud_cover': 90, 'temperature': 18, 'humidity': 85}
        }
        
        fig = go.Figure()
        
        colors = ['#ff6b35', '#feca57', '#48dbfb', '#0abde3']
        
        for i, (scenario, conditions) in enumerate(weather_scenarios.items()):
            irradiance_values = []
            for hour in hours:
                irradiance = self._calculate_solar_irradiance(
                    latitude, longitude, day_of_year, hour, conditions
                )
                irradiance_values.append(irradiance)
            
            fig.add_trace(go.Scatter(
                x=hours,
                y=irradiance_values,
                mode='lines',
                name=scenario,
                line=dict(color=colors[i], width=3)
            ))
        
        fig.update_layout(
            title=f'Weather Impact on Solar Irradiance - {date.strftime("%B %d, %Y")}',
            xaxis_title='Hour of Day',
            yaxis_title='Solar Irradiance (W/m²)',
            showlegend=True,
            width=800,
            height=500
        )
        
        return fig
    
    def create_solar_path_visualization(self, latitude, longitude, date=None):
        """
        Create solar path visualization showing sun's trajectory.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            date: Date for visualization
            
        Returns:
            plotly.graph_objects.Figure: Solar path visualization
        """
        
        if date is None:
            date = datetime.now()
        
        day_of_year = date.timetuple().tm_yday
        
        # Generate solar path data
        hours = np.arange(0, 24, 0.1)
        altitudes = []
        azimuths = []
        
        for hour in hours:
            # Calculate solar altitude and azimuth
            lat_rad = math.radians(latitude)
            declination = 23.45 * math.sin(math.radians(360/365 * (day_of_year - 80)))
            declination_rad = math.radians(declination)
            
            solar_noon = 12
            hour_angle = 15 * (hour - solar_noon)
            hour_angle_rad = math.radians(hour_angle)
            
            # Calculate altitude
            sin_altitude = (math.sin(lat_rad) * math.sin(declination_rad) + 
                           math.cos(lat_rad) * math.cos(declination_rad) * math.cos(hour_angle_rad))
            altitude = math.asin(max(-1, min(1, sin_altitude)))
            
            # Calculate azimuth
            cos_azimuth = ((math.sin(declination_rad) - math.sin(altitude) * math.sin(lat_rad)) / 
                          (math.cos(altitude) * math.cos(lat_rad)))
            azimuth = math.acos(max(-1, min(1, cos_azimuth)))
            
            if hour_angle < 0:
                azimuth = 2 * math.pi - azimuth
            
            altitudes.append(math.degrees(altitude))
            azimuths.append(math.degrees(azimuth))
        
        # Create polar plot
        fig = go.Figure()
        
        # Add solar path
        fig.add_trace(go.Scatterpolar(
            r=altitudes,
            theta=azimuths,
            mode='lines',
            name='Solar Path',
            line=dict(color='#ff6b35', width=3)
        ))
        
        # Add key points
        key_hours = [6, 9, 12, 15, 18]
        key_altitudes = []
        key_azimuths = []
        key_labels = []
        
        for hour in key_hours:
            if 6 <= hour <= 18:
                idx = int(hour * 10)  # Convert to index
                if idx < len(altitudes):
                    key_altitudes.append(altitudes[idx])
                    key_azimuths.append(azimuths[idx])
                    key_labels.append(f'{hour}:00')
        
        fig.add_trace(go.Scatterpolar(
            r=key_altitudes,
            theta=key_azimuths,
            mode='markers+text',
            name='Key Times',
            text=key_labels,
            textposition='middle center',
            marker=dict(color='#ffd93d', size=10, symbol='diamond'),
            textfont=dict(size=12, color='black')
        ))
        
        fig.update_layout(
            title=f'Solar Path - {date.strftime("%B %d, %Y")} - Lat: {latitude}°',
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 90],
                    ticktext=['0°', '30°', '60°', '90°'],
                    tickvals=[0, 30, 60, 90]
                ),
                angularaxis=dict(
                    visible=True,
                    ticktext=['N', 'E', 'S', 'W'],
                    tickvals=[0, 90, 180, 270]
                )
            ),
            showlegend=True,
            width=600,
            height=600
        )
        
        return fig

def main():
    """Main function for testing the visualizer."""
    visualizer = Solar3DVisualizer()
    
    # Example usage
    latitude = 40.7128  # New York
    longitude = -74.0060
    
    # Create visualizations
    fig_3d = visualizer.create_3d_irradiance_map(latitude, longitude)
    fig_daily = visualizer.create_daily_irradiance_profile(latitude, longitude)
    fig_seasonal = visualizer.create_seasonal_comparison(latitude, longitude)
    fig_weather = visualizer.create_weather_impact_analysis(latitude, longitude)
    fig_solar_path = visualizer.create_solar_path_visualization(latitude, longitude)
    
    print("3D Solar Irradiance Visualizations created successfully!")
    print("Use these figures in your Streamlit app or save them as HTML files.")

if __name__ == "__main__":
    main()