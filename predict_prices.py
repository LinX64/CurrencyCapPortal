#!/usr/bin/env python3
"""
Generate price predictions using trained models.

Usage:
    python predict_prices.py --currency usd --hours 24
    python predict_prices.py --all --hours 48
"""

import argparse
import json
import os
from datetime import datetime
from ml.predictor import CurrencyPredictor


def predict_single_currency(currency_code: str, hours: int = 24, save: bool = True):
    """Generate predictions for a single currency."""
    print(f"Generating {hours}h predictions for {currency_code.upper()}...")

    predictor = CurrencyPredictor(currency_code=currency_code)

    try:
        predictions = predictor.predict_future(hours=hours)

        if save:
            os.makedirs('api/predictions', exist_ok=True)
            output_path = f'api/predictions/{currency_code}.json'

            model_info = predictor.get_model_info()

            output = {
                'currency_code': currency_code,
                'generated_at': datetime.now().isoformat(),
                'prediction_horizon_hours': hours,
                'model_info': {
                    'trained_at': model_info.get('trained_at'),
                    'test_mae': model_info['metrics']['mae'],
                    'test_mape': model_info['metrics']['mape']
                },
                'predictions': predictions
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            print(f"✓ Saved predictions to {output_path}")

        return predictions
    except Exception as e:
        print(f"✗ Failed to generate predictions for {currency_code.upper()}: {e}")
        return None


def predict_all_currencies(hours: int = 24):
    """Generate predictions for all currencies with trained models."""
    model_dir = 'models'
    if not os.path.exists(model_dir):
        print("No models directory found. Train models first.")
        return

    model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.keras')]
    currencies = [f.replace('_model.keras', '') for f in model_files]

    if not currencies:
        print("No trained models found. Train models first using train_model.py")
        return

    print(f"Found {len(currencies)} trained models")

    results = {}
    for currency in currencies:
        predictions = predict_single_currency(currency, hours=hours, save=True)
        results[currency] = predictions is not None

    print(f"\n{'='*60}")
    print("Prediction Summary")
    print(f"{'='*60}")
    successful = sum(1 for v in results.values() if v)
    print(f"Successful: {successful}/{len(results)}")

    os.makedirs('api/predictions', exist_ok=True)
    index_path = 'api/predictions/index.json'
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'currencies': list(results.keys()),
            'prediction_horizon_hours': hours
        }, f, indent=2)

    print(f"\n✓ Updated predictions index at {index_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate currency price predictions')
    parser.add_argument('--currency', type=str, help='Currency code to predict')
    parser.add_argument('--all', action='store_true', help='Predict for all trained models')
    parser.add_argument('--hours', type=int, default=24, help='Prediction horizon in hours')

    args = parser.parse_args()

    if args.all:
        predict_all_currencies(hours=args.hours)
    elif args.currency:
        predict_single_currency(args.currency, hours=args.hours, save=True)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
