#!/usr/bin/env python3
"""
ROI Calculator for Solar Installations
Calculates return on investment, payback period, and financial benefits of solar installations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json

class SolarROICalculator:
    def __init__(self):
        """Initialize the ROI calculator with default parameters."""
        self.default_params = {
            'installation_cost_per_kw': 2500,  # USD per kW
            'maintenance_cost_percent': 0.5,    # 0.5% of installation cost per year
            'inverter_replacement_cost': 2000,  # USD
            'inverter_lifespan': 15,            # years
            'panel_lifespan': 25,               # years
            'degradation_rate': 0.5,            # 0.5% per year
            'electricity_price_inflation': 2.5,  # 2.5% per year
            'discount_rate': 5.0,               # 5% discount rate for NPV
            'tax_credit_percent': 30,           # 30% federal tax credit
            'state_tax_credit_percent': 0,      # Varies by state
            'net_metering_rate': 0.8,           # 80% of retail rate for excess generation
        }
        
        # Regional electricity prices (USD per kWh)
        self.regional_prices = {
            'US_National': 0.14,
            'California': 0.22,
            'Texas': 0.12,
            'New_York': 0.18,
            'Florida': 0.12,
            'Arizona': 0.13,
            'Colorado': 0.12,
            'Washington': 0.10,
            'Oregon': 0.11,
            'Nevada': 0.12,
            'Utah': 0.11,
            'New_Mexico': 0.13,
            'Europe_UK': 0.28,
            'Europe_Germany': 0.32,
            'Europe_France': 0.18,
            'Asia_Japan': 0.25,
            'Asia_China': 0.08,
            'Asia_India': 0.10,
            'Australia': 0.20,
            'Canada': 0.12
        }
    
    def calculate_roi(self, system_size_kw, daily_generation_kwh, region='US_National', 
                     custom_params=None, years=25):
        """
        Calculate comprehensive ROI analysis for solar installation.
        
        Args:
            system_size_kw: Solar system size in kW
            daily_generation_kwh: Average daily generation in kWh
            region: Geographic region for electricity pricing
            custom_params: Custom parameters to override defaults
            years: Analysis period in years
            
        Returns:
            dict: Comprehensive ROI analysis results
        """
        
        # Merge custom parameters with defaults
        params = self.default_params.copy()
        if custom_params:
            params.update(custom_params)
        
        # Get electricity price for region
        electricity_price = self.regional_prices.get(region, self.regional_prices['US_National'])
        
        # Calculate annual generation
        annual_generation_kwh = daily_generation_kwh * 365
        
        # Calculate installation costs
        total_installation_cost = system_size_kw * params['installation_cost_per_kw']
        
        # Apply tax credits
        federal_tax_credit = total_installation_cost * (params['tax_credit_percent'] / 100)
        state_tax_credit = total_installation_cost * (params['state_tax_credit_percent'] / 100)
        net_installation_cost = total_installation_cost - federal_tax_credit - state_tax_credit
        
        # Calculate annual cash flows
        annual_cash_flows = []
        cumulative_savings = 0
        payback_period = None
        
        for year in range(1, years + 1):
            # Calculate generation with degradation
            degradation_factor = (1 - params['degradation_rate'] / 100) ** (year - 1)
            year_generation = annual_generation_kwh * degradation_factor
            
            # Calculate electricity savings
            electricity_price_year = electricity_price * (1 + params['electricity_price_inflation'] / 100) ** (year - 1)
            electricity_savings = year_generation * electricity_price_year
            
            # Calculate maintenance costs
            maintenance_cost = total_installation_cost * (params['maintenance_cost_percent'] / 100)
            
            # Add inverter replacement cost
            inverter_cost = 0
            if year % params['inverter_lifespan'] == 0 and year > 0:
                inverter_cost = params['inverter_replacement_cost']
            
            # Net annual cash flow
            net_cash_flow = electricity_savings - maintenance_cost - inverter_cost
            annual_cash_flows.append(net_cash_flow)
            
            # Check payback period
            cumulative_savings += net_cash_flow
            if payback_period is None and cumulative_savings >= net_installation_cost:
                payback_period = year
        
        # Calculate financial metrics
        total_savings = sum(annual_cash_flows)
        net_profit = total_savings - net_installation_cost
        roi_percent = (net_profit / net_installation_cost) * 100 if net_installation_cost > 0 else 0
        
        # Calculate NPV
        npv = -net_installation_cost
        for i, cash_flow in enumerate(annual_cash_flows):
            npv += cash_flow / ((1 + params['discount_rate'] / 100) ** (i + 1))
        
        # Calculate IRR (simplified approximation)
        irr_approx = self._calculate_irr_approximation(net_installation_cost, annual_cash_flows)
        
        # Calculate levelized cost of energy (LCOE)
        total_cost = net_installation_cost + sum([cf for cf in annual_cash_flows if cf < 0])
        total_energy = sum([annual_generation_kwh * (1 - params['degradation_rate'] / 100) ** year 
                           for year in range(years)])
        lcoe = total_cost / total_energy if total_energy > 0 else 0
        
        # Calculate environmental benefits
        co2_saved_kg = total_energy * 0.85  # 0.85 kg CO2 per kWh (grid average)
        trees_equivalent = co2_saved_kg / 22  # 22 kg CO2 absorbed by one tree per year
        
        # Create detailed results
        results = {
            'system_info': {
                'system_size_kw': system_size_kw,
                'daily_generation_kwh': daily_generation_kwh,
                'annual_generation_kwh': annual_generation_kwh,
                'region': region,
                'electricity_price_usd_per_kwh': electricity_price
            },
            'costs': {
                'total_installation_cost': round(total_installation_cost, 2),
                'federal_tax_credit': round(federal_tax_credit, 2),
                'state_tax_credit': round(state_tax_credit, 2),
                'net_installation_cost': round(net_installation_cost, 2)
            },
            'financial_metrics': {
                'total_savings': round(total_savings, 2),
                'net_profit': round(net_profit, 2),
                'roi_percent': round(roi_percent, 2),
                'payback_period_years': payback_period,
                'npv': round(npv, 2),
                'irr_approx_percent': round(irr_approx, 2),
                'lcoe_usd_per_kwh': round(lcoe, 3)
            },
            'environmental_benefits': {
                'co2_saved_kg': round(co2_saved_kg, 0),
                'trees_equivalent': round(trees_equivalent, 0),
                'cars_off_road': round(co2_saved_kg / 4600, 1)  # 4600 kg CO2 per car per year
            },
            'annual_breakdown': [
                {
                    'year': year + 1,
                    'generation_kwh': round(annual_generation_kwh * (1 - params['degradation_rate'] / 100) ** year, 0),
                    'electricity_savings': round(annual_cash_flows[year], 2),
                    'maintenance_cost': round(total_installation_cost * (params['maintenance_cost_percent'] / 100), 2),
                    'net_cash_flow': round(annual_cash_flows[year], 2)
                }
                for year in range(min(10, years))  # Show first 10 years
            ],
            'parameters_used': params,
            'analysis_period_years': years
        }
        
        return results
    
    def _calculate_irr_approximation(self, initial_investment, cash_flows):
        """Calculate approximate IRR using simplified method."""
        try:
            total_cash_flow = sum(cash_flows)
            avg_annual_cash_flow = total_cash_flow / len(cash_flows)
            
            # Simple approximation: (avg annual return / initial investment) * 100
            irr_approx = (avg_annual_cash_flow / initial_investment) * 100
            return max(0, irr_approx)
        except:
            return 0
    
    def get_regional_prices(self):
        """Get available regional electricity prices."""
        return self.regional_prices
    
    def calculate_optimal_system_size(self, daily_consumption_kwh, region='US_National', 
                                    roof_area_sqm=None, budget_usd=None):
        """
        Calculate optimal solar system size based on consumption and constraints.
        
        Args:
            daily_consumption_kwh: Average daily electricity consumption
            region: Geographic region
            roof_area_sqm: Available roof area in square meters
            budget_usd: Maximum budget in USD
            
        Returns:
            dict: Optimal system recommendations
        """
        
        # Calculate system size to cover 100% of consumption
        target_daily_generation = daily_consumption_kwh
        target_system_size = target_daily_generation / 4  # Assume 4 kWh per kW per day average
        
        # Apply constraints
        max_system_size = target_system_size
        
        if roof_area_sqm:
            # Assume 6.5 kW per 100 sqm (typical solar panel efficiency)
            roof_max_size = (roof_area_sqm / 100) * 6.5
            max_system_size = min(max_system_size, roof_max_size)
        
        if budget_usd:
            # Calculate maximum system size based on budget
            budget_max_size = budget_usd / self.default_params['installation_cost_per_kw']
            max_system_size = min(max_system_size, budget_max_size)
        
        # Calculate ROI for different system sizes
        system_options = []
        for size in [max_system_size * 0.5, max_system_size * 0.75, max_system_size]:
            if size > 0:
                daily_gen = size * 4  # Estimate daily generation
                roi_result = self.calculate_roi(size, daily_gen, region)
                system_options.append({
                    'system_size_kw': round(size, 2),
                    'daily_generation_kwh': round(daily_gen, 1),
                    'installation_cost': roi_result['costs']['net_installation_cost'],
                    'roi_percent': roi_result['financial_metrics']['roi_percent'],
                    'payback_years': roi_result['financial_metrics']['payback_period_years'],
                    'annual_savings': round(roi_result['annual_breakdown'][0]['electricity_savings'], 2)
                })
        
        return {
            'target_consumption_kwh': daily_consumption_kwh,
            'recommended_system_size_kw': round(max_system_size, 2),
            'system_options': system_options,
            'constraints': {
                'roof_area_sqm': roof_area_sqm,
                'budget_usd': budget_usd,
                'max_system_size_kw': round(max_system_size, 2)
            }
        }
    
    def compare_financing_options(self, system_size_kw, daily_generation_kwh, region='US_National'):
        """
        Compare different financing options for solar installation.
        
        Args:
            system_size_kw: Solar system size in kW
            daily_generation_kwh: Average daily generation in kWh
            region: Geographic region
            
        Returns:
            dict: Comparison of financing options
        """
        
        # Calculate cash purchase scenario
        cash_roi = self.calculate_roi(system_size_kw, daily_generation_kwh, region)
        
        # Calculate loan scenario (7% interest, 20 years)
        loan_params = self.default_params.copy()
        loan_params['installation_cost_per_kw'] = 0  # No upfront cost
        loan_roi = self.calculate_roi(system_size_kw, daily_generation_kwh, region, loan_params)
        
        # Add loan payments to loan scenario
        total_loan_amount = cash_roi['costs']['net_installation_cost']
        monthly_payment = (total_loan_amount * 0.07 / 12) / (1 - (1 + 0.07 / 12) ** (-20 * 12))
        annual_loan_payment = monthly_payment * 12
        
        # Adjust loan cash flows
        for year in range(20):
            if year < len(loan_roi['annual_breakdown']):
                loan_roi['annual_breakdown'][year]['net_cash_flow'] -= annual_loan_payment
        
        # Calculate lease scenario (no upfront cost, monthly lease payment)
        lease_monthly = 150  # Example lease payment
        lease_annual = lease_monthly * 12
        
        lease_params = self.default_params.copy()
        lease_params['installation_cost_per_kw'] = 0
        lease_roi = self.calculate_roi(system_size_kw, daily_generation_kwh, region, lease_params)
        
        # Adjust lease cash flows
        for year in range(25):
            if year < len(lease_roi['annual_breakdown']):
                lease_roi['annual_breakdown'][year]['net_cash_flow'] -= lease_annual
        
        return {
            'cash_purchase': {
                'upfront_cost': cash_roi['costs']['net_installation_cost'],
                'roi_percent': cash_roi['financial_metrics']['roi_percent'],
                'payback_years': cash_roi['financial_metrics']['payback_period_years'],
                'total_savings': cash_roi['financial_metrics']['total_savings']
            },
            'loan_financing': {
                'upfront_cost': 0,
                'monthly_payment': round(monthly_payment, 2),
                'total_loan_cost': round(monthly_payment * 12 * 20, 2),
                'roi_percent': round((cash_roi['financial_metrics']['total_savings'] - monthly_payment * 12 * 20) / 0 * 100, 2) if 0 > 0 else 0,
                'payback_years': None  # Loan scenario doesn't have traditional payback
            },
            'lease_option': {
                'upfront_cost': 0,
                'monthly_payment': lease_monthly,
                'total_lease_cost': lease_annual * 25,
                'roi_percent': round((cash_roi['financial_metrics']['total_savings'] - lease_annual * 25) / 0 * 100, 2) if 0 > 0 else 0,
                'payback_years': None  # Lease scenario doesn't have traditional payback
            }
        }

if __name__ == "__main__":
    # Test the ROI calculator
    calculator = SolarROICalculator()
    
    # Example calculation
    result = calculator.calculate_roi(
        system_size_kw=5.0,
        daily_generation_kwh=20.0,
        region='California',
        years=25
    )
    
    print("Solar ROI Analysis Results:")
    print(f"System Size: {result['system_info']['system_size_kw']} kW")
    print(f"Daily Generation: {result['system_info']['daily_generation_kwh']} kWh")
    print(f"Installation Cost: ${result['costs']['net_installation_cost']:,.2f}")
    print(f"ROI: {result['financial_metrics']['roi_percent']:.1f}%")
    print(f"Payback Period: {result['financial_metrics']['payback_period_years']} years")
    print(f"Total Savings: ${result['financial_metrics']['total_savings']:,.2f}")
    print(f"CO2 Saved: {result['environmental_benefits']['co2_saved_kg']:,.0f} kg")