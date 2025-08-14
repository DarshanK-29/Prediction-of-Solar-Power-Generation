#!/usr/bin/env python3
"""
Solar Power Generation Prediction - Training Pipeline
This script runs the complete training pipeline including data generation and model training.
"""

import os
import sys
from data_generator import generate_solar_power_data
from model_trainer import SolarPowerModelTrainer

def main():
    """Run the complete training pipeline."""
    print("=" * 60)
    print("☀️ Solar Power Generation Prediction - Training Pipeline")
    print("=" * 60)
    
    # Step 1: Generate dataset
    print("\n📊 Step 1: Generating synthetic dataset...")
    try:
        df = generate_solar_power_data(10000)
        df.to_csv('augmented_solarpowergeneration.csv', index=False)
        print(f"✅ Dataset generated successfully with {len(df)} samples")
        print(f"   Features: {list(df.columns)}")
        print(f"   Target: power_generated")
        print(f"   Data range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Power range: {df['power_generated'].min():.2f} to {df['power_generated'].max():.2f} kW")
    except Exception as e:
        print(f"❌ Error generating dataset: {e}")
        sys.exit(1)
    
    # Step 2: Train and evaluate models
    print("\n🤖 Step 2: Training and evaluating models...")
    try:
        trainer = SolarPowerModelTrainer()
        best_model = trainer.train_and_evaluate()
        print(f"✅ Training completed successfully!")
        print(f"   Best model: {trainer.best_model_name}")
        print(f"   Model saved as: solar_power_model.pkl")
        print(f"   Feature importance saved as: feature_importance.csv")
        print(f"   Evaluation plots saved as: model_evaluation_results.png")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        sys.exit(1)
    
    # Step 3: Verify files
    print("\n🔍 Step 3: Verifying generated files...")
    required_files = [
        'augmented_solarpowergeneration.csv',
        'solar_power_model.pkl',
        'feature_importance.csv',
        'model_evaluation_results.png'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (missing)")
    
    print("\n" + "=" * 60)
    print("🎉 Training pipeline completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Get your OpenWeatherMap API key from: https://openweathermap.org/api")
    print("2. Copy .env.example to .env and add your API key")
    print("3. Run the Streamlit app: streamlit run app.py")
    print("\nHappy predicting! ☀️")

if __name__ == "__main__":
    main()