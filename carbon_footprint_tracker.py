#!/usr/bin/env python3
"""
Carbon Footprint Reduction Tracker
Calculates CO2 emissions saved through solar generation and environmental impact metrics.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class CarbonFootprintTracker:
    def __init__(self):
        """Initialize the carbon footprint tracker."""
        
        # CO2 emission factors (kg CO2 per kWh)
        self.emission_factors = {
            'grid_average': 0.85,      # Average grid electricity
            'coal': 1.0,               # Coal-fired power
            'natural_gas': 0.5,        # Natural gas power
            'nuclear': 0.0,            # Nuclear power (no CO2)
            'hydro': 0.0,              # Hydroelectric (no CO2)
            'wind': 0.0,               # Wind power (no CO2)
            'solar': 0.0,              # Solar power (no CO2)
            'biomass': 0.2,            # Biomass power
            'geothermal': 0.0,         # Geothermal (no CO2)
            'oil': 0.7                 # Oil-fired power
        }
        
        # Regional grid mixes (percentage of each source)
        self.regional_grid_mixes = {
            'US_National': {
                'coal': 20, 'natural_gas': 40, 'nuclear': 20, 'hydro': 7, 'wind': 8, 'solar': 3, 'other': 2
            },
            'California': {
                'coal': 0, 'natural_gas': 45, 'nuclear': 9, 'hydro': 12, 'wind': 12, 'solar': 20, 'other': 2
            },
            'Texas': {
                'coal': 18, 'natural_gas': 52, 'nuclear': 8, 'hydro': 1, 'wind': 20, 'solar': 1, 'other': 0
            },
            'New_York': {
                'coal': 1, 'natural_gas': 40, 'nuclear': 30, 'hydro': 20, 'wind': 4, 'solar': 3, 'other': 2
            },
            'Europe_UK': {
                'coal': 2, 'natural_gas': 40, 'nuclear': 15, 'hydro': 2, 'wind': 25, 'solar': 4, 'other': 12
            },
            'Europe_Germany': {
                'coal': 25, 'natural_gas': 15, 'nuclear': 6, 'hydro': 4, 'wind': 25, 'solar': 10, 'other': 15
            },
            'Asia_China': {
                'coal': 60, 'natural_gas': 8, 'nuclear': 5, 'hydro': 18, 'wind': 6, 'solar': 2, 'other': 1
            },
            'Asia_Japan': {
                'coal': 30, 'natural_gas': 35, 'nuclear': 5, 'hydro': 8, 'wind': 1, 'solar': 8, 'other': 13
            },
            'Australia': {
                'coal': 60, 'natural_gas': 20, 'nuclear': 0, 'hydro': 7, 'wind': 8, 'solar': 3, 'other': 2
            }
        }
        
        # Environmental equivalencies
        self.environmental_equivalencies = {
            'trees_planted': 22,       # kg CO2 absorbed by one tree per year
            'cars_off_road': 4600,     # kg CO2 per car per year
            'homes_powered': 12000,    # kWh per home per year
            'smartphones_charged': 0.05, # kWh per smartphone charge
            'lightbulb_hours': 0.1,    # kWh per hour for LED bulb
            'flight_km': 0.255,        # kg CO2 per km flown
            'gasoline_liters': 2.31,   # kg CO2 per liter of gasoline
            'beef_kg': 13.3,           # kg CO2 per kg of beef
            'plastic_bags': 0.1,       # kg CO2 per plastic bag
            'paper_sheets': 0.004      # kg CO2 per sheet of paper
        }
    
    def calculate_carbon_savings(self, solar_generation_kwh, region='US_National', 
                                time_period='daily', custom_grid_mix=None):
        """
        Calculate carbon savings from solar generation.
        
        Args:
            solar_generation_kwh: Solar generation in kWh
            region: Geographic region for grid mix
            time_period: Time period for calculation ('daily', 'monthly', 'yearly')
            custom_grid_mix: Custom grid mix percentages
            
        Returns:
            dict: Carbon savings and environmental impact
        """
        
        # Get grid mix for region
        if custom_grid_mix:
            grid_mix = custom_grid_mix
        else:
            grid_mix = self.regional_grid_mixes.get(region, self.regional_grid_mixes['US_National'])
        
        # Calculate weighted emission factor
        weighted_emission_factor = 0
        for source, percentage in grid_mix.items():
            if source in self.emission_factors:
                weighted_emission_factor += (percentage / 100) * self.emission_factors[source]
        
        # Calculate CO2 emissions avoided
        co2_saved_kg = solar_generation_kwh * weighted_emission_factor
        
        # Calculate environmental equivalencies
        environmental_impact = self._calculate_environmental_equivalencies(co2_saved_kg)
        
        # Calculate time-based metrics
        time_metrics = self._calculate_time_metrics(solar_generation_kwh, time_period)
        
        return {
            'solar_generation_kwh': round(solar_generation_kwh, 2),
            'region': region,
            'grid_emission_factor': round(weighted_emission_factor, 3),
            'co2_saved_kg': round(co2_saved_kg, 2),
            'time_period': time_period,
            'environmental_impact': environmental_impact,
            'time_metrics': time_metrics,
            'grid_mix_used': grid_mix,
            'calculation_date': datetime.now().isoformat()
        }
    
    def _calculate_environmental_equivalencies(self, co2_saved_kg):
        """Calculate environmental equivalencies for CO2 savings."""
        
        equivalencies = {}
        
        for metric, factor in self.environmental_equivalencies.items():
            if factor > 0:
                equivalencies[metric] = round(co2_saved_kg / factor, 2)
            else:
                equivalencies[metric] = 0
        
        return equivalencies
    
    def _calculate_time_metrics(self, solar_generation_kwh, time_period):
        """Calculate time-based environmental metrics."""
        
        # Calculate annualized metrics
        if time_period == 'daily':
            annual_generation = solar_generation_kwh * 365
        elif time_period == 'monthly':
            annual_generation = solar_generation_kwh * 12
        else:  # yearly
            annual_generation = solar_generation_kwh
        
        # Calculate annual CO2 savings
        annual_co2_saved = annual_generation * self.emission_factors['grid_average']
        
        # Calculate long-term impact (25 years)
        lifetime_co2_saved = annual_co2_saved * 25
        
        return {
            'annual_generation_kwh': round(annual_generation, 2),
            'annual_co2_saved_kg': round(annual_co2_saved, 2),
            'lifetime_co2_saved_kg': round(lifetime_co2_saved, 2),
            'lifetime_trees_equivalent': round(lifetime_co2_saved / self.environmental_equivalencies['trees_planted'], 0),
            'lifetime_cars_equivalent': round(lifetime_co2_saved / self.environmental_equivalencies['cars_off_road'], 1)
        }
    
    def track_daily_carbon_savings(self, daily_generations, region='US_National'):
        """
        Track daily carbon savings over time.
        
        Args:
            daily_generations: List of daily generation values (kWh)
            region: Geographic region
            
        Returns:
            dict: Daily tracking data
        """
        
        tracking_data = []
        cumulative_co2_saved = 0
        
        for i, generation in enumerate(daily_generations):
            # Calculate daily savings
            daily_savings = self.calculate_carbon_savings(generation, region, 'daily')
            
            cumulative_co2_saved += daily_savings['co2_saved_kg']
            
            tracking_data.append({
                'day': i + 1,
                'generation_kwh': generation,
                'co2_saved_kg': daily_savings['co2_saved_kg'],
                'cumulative_co2_saved_kg': round(cumulative_co2_saved, 2),
                'trees_equivalent': daily_savings['environmental_impact']['trees_planted'],
                'cars_equivalent': daily_savings['environmental_impact']['cars_off_road']
            })
        
        return {
            'tracking_data': tracking_data,
            'total_generation_kwh': sum(daily_generations),
            'total_co2_saved_kg': round(cumulative_co2_saved, 2),
            'average_daily_co2_saved_kg': round(cumulative_co2_saved / len(daily_generations), 2),
            'region': region,
            'period_days': len(daily_generations)
        }
    
    def compare_regional_impact(self, solar_generation_kwh):
        """
        Compare environmental impact across different regions.
        
        Args:
            solar_generation_kwh: Solar generation in kWh
            
        Returns:
            dict: Regional comparison data
        """
        
        regional_comparison = {}
        
        for region in self.regional_grid_mixes.keys():
            savings = self.calculate_carbon_savings(solar_generation_kwh, region)
            regional_comparison[region] = {
                'co2_saved_kg': savings['co2_saved_kg'],
                'grid_emission_factor': savings['grid_emission_factor'],
                'trees_equivalent': savings['environmental_impact']['trees_planted'],
                'cars_equivalent': savings['environmental_impact']['cars_off_road']
            }
        
        return regional_comparison
    
    def create_carbon_savings_visualization(self, tracking_data):
        """
        Create visualization of carbon savings over time.
        
        Args:
            tracking_data: Data from track_daily_carbon_savings
            
        Returns:
            plotly.graph_objects.Figure: Carbon savings visualization
        """
        
        df = pd.DataFrame(tracking_data['tracking_data'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Daily CO2 Savings', 'Cumulative CO2 Savings', 
                           'Daily Generation', 'Environmental Impact'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Daily CO2 savings
        fig.add_trace(
            go.Scatter(
                x=df['day'],
                y=df['co2_saved_kg'],
                mode='lines+markers',
                name='Daily CO2 Saved',
                line=dict(color='#2ecc71', width=2)
            ),
            row=1, col=1
        )
        
        # Cumulative CO2 savings
        fig.add_trace(
            go.Scatter(
                x=df['day'],
                y=df['cumulative_co2_saved_kg'],
                mode='lines',
                name='Cumulative CO2 Saved',
                line=dict(color='#e74c3c', width=3)
            ),
            row=1, col=2
        )
        
        # Daily generation
        fig.add_trace(
            go.Bar(
                x=df['day'],
                y=df['generation_kwh'],
                name='Daily Generation',
                marker_color='#3498db'
            ),
            row=2, col=1
        )
        
        # Environmental impact (trees equivalent)
        fig.add_trace(
            go.Scatter(
                x=df['day'],
                y=df['trees_equivalent'],
                mode='lines',
                name='Trees Equivalent',
                line=dict(color='#27ae60', width=2)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Carbon Footprint Reduction Tracking - {tracking_data['period_days']} Days",
            showlegend=False,
            width=1000,
            height=600
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Day", row=1, col=1)
        fig.update_yaxes(title_text="CO2 Saved (kg)", row=1, col=1)
        fig.update_xaxes(title_text="Day", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative CO2 Saved (kg)", row=1, col=2)
        fig.update_xaxes(title_text="Day", row=2, col=1)
        fig.update_yaxes(title_text="Generation (kWh)", row=2, col=1)
        fig.update_xaxes(title_text="Day", row=2, col=2)
        fig.update_yaxes(title_text="Trees Equivalent", row=2, col=2)
        
        return fig
    
    def create_regional_comparison_chart(self, regional_comparison):
        """
        Create visualization comparing regional carbon savings.
        
        Args:
            regional_comparison: Data from compare_regional_impact
            
        Returns:
            plotly.graph_objects.Figure: Regional comparison chart
        """
        
        regions = list(regional_comparison.keys())
        co2_savings = [regional_comparison[region]['co2_saved_kg'] for region in regions]
        emission_factors = [regional_comparison[region]['grid_emission_factor'] for region in regions]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['CO2 Savings by Region', 'Grid Emission Factors'],
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # CO2 savings by region
        fig.add_trace(
            go.Bar(
                x=regions,
                y=co2_savings,
                name='CO2 Saved (kg)',
                marker_color='#2ecc71'
            ),
            row=1, col=1
        )
        
        # Grid emission factors
        fig.add_trace(
            go.Bar(
                x=regions,
                y=emission_factors,
                name='Grid Emission Factor',
                marker_color='#e74c3c'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Regional Carbon Savings Comparison",
            showlegend=False,
            width=1000,
            height=400
        )
        
        fig.update_xaxes(title_text="Region", row=1, col=1)
        fig.update_yaxes(title_text="CO2 Saved (kg)", row=1, col=1)
        fig.update_xaxes(title_text="Region", row=1, col=2)
        fig.update_yaxes(title_text="Emission Factor (kg CO2/kWh)", row=1, col=2)
        
        return fig
    
    def calculate_system_efficiency_metrics(self, system_size_kw, actual_generation_kwh, 
                                          time_period_days=30):
        """
        Calculate system efficiency and carbon impact metrics.
        
        Args:
            system_size_kw: Solar system size in kW
            actual_generation_kwh: Actual generation over time period
            time_period_days: Time period in days
            
        Returns:
            dict: Efficiency and carbon metrics
        """
        
        # Calculate theoretical generation
        theoretical_daily_generation = system_size_kw * 4  # 4 kWh per kW per day average
        theoretical_total_generation = theoretical_daily_generation * time_period_days
        
        # Calculate efficiency
        efficiency_percent = (actual_generation_kwh / theoretical_total_generation) * 100
        
        # Calculate carbon savings
        carbon_savings = self.calculate_carbon_savings(actual_generation_kwh)
        
        # Calculate efficiency-adjusted metrics
        if efficiency_percent > 0:
            efficiency_adjusted_co2 = carbon_savings['co2_saved_kg'] / (efficiency_percent / 100)
        else:
            efficiency_adjusted_co2 = 0
        
        return {
            'system_size_kw': system_size_kw,
            'actual_generation_kwh': actual_generation_kwh,
            'theoretical_generation_kwh': theoretical_total_generation,
            'efficiency_percent': round(efficiency_percent, 2),
            'time_period_days': time_period_days,
            'carbon_savings': carbon_savings,
            'efficiency_adjusted_co2_kg': round(efficiency_adjusted_co2, 2),
            'performance_rating': self._get_performance_rating(efficiency_percent)
        }
    
    def _get_performance_rating(self, efficiency_percent):
        """Get performance rating based on efficiency."""
        
        if efficiency_percent >= 90:
            return 'Excellent'
        elif efficiency_percent >= 80:
            return 'Very Good'
        elif efficiency_percent >= 70:
            return 'Good'
        elif efficiency_percent >= 60:
            return 'Fair'
        else:
            return 'Poor'

def main():
    """Test the carbon footprint tracker."""
    
    tracker = CarbonFootprintTracker()
    
    print("Testing Carbon Footprint Tracker...")
    print("=" * 50)
    
    # Test daily carbon savings
    print("1. Calculating daily carbon savings...")
    daily_savings = tracker.calculate_carbon_savings(20.0, 'California', 'daily')
    print(f"   Daily Generation: {daily_savings['solar_generation_kwh']} kWh")
    print(f"   CO2 Saved: {daily_savings['co2_saved_kg']} kg")
    print(f"   Trees Equivalent: {daily_savings['environmental_impact']['trees_planted']}")
    print(f"   Cars Equivalent: {daily_savings['environmental_impact']['cars_off_road']}")
    
    # Test tracking over time
    print("\n2. Tracking carbon savings over time...")
    daily_generations = [15, 18, 22, 19, 25, 20, 17]  # 7 days of data
    tracking_data = tracker.track_daily_carbon_savings(daily_generations, 'California')
    print(f"   Total CO2 Saved: {tracking_data['total_co2_saved_kg']} kg")
    print(f"   Average Daily CO2 Saved: {tracking_data['average_daily_co2_saved_kg']} kg")
    
    # Test regional comparison
    print("\n3. Comparing regional impact...")
    regional_comparison = tracker.compare_regional_impact(20.0)
    print("   Regional CO2 Savings (kg):")
    for region, data in regional_comparison.items():
        print(f"     {region}: {data['co2_saved_kg']}")
    
    # Test system efficiency
    print("\n4. Calculating system efficiency...")
    efficiency_metrics = tracker.calculate_system_efficiency_metrics(5.0, 140.0, 7)
    print(f"   System Efficiency: {efficiency_metrics['efficiency_percent']}%")
    print(f"   Performance Rating: {efficiency_metrics['performance_rating']}")
    
    print("\n✅ Carbon footprint tracker test completed!")

if __name__ == "__main__":
    main()