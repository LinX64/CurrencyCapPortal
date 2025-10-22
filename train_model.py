#!/usr/bin/env python3
"""
Train ML models for currency price prediction.

Usage:
    python train_model.py --currency usd
    python train_model.py --currency eur --epochs 150
    python train_model.py --all
"""

import argparse
import json
from ml.predictor import CurrencyPredictor


def train_single_currency(currency_code: str, epochs: int = 100):
    """Train model for a single currency."""
    print(f"\n{'='*60}")
    print(f"Training model for {currency_code.upper()}")
    print(f"{'='*60}\n")

    predictor = CurrencyPredictor(currency_code=currency_code)

    try:
        metrics = predictor.train_model(epochs=epochs)
        print(f"\n✓ Successfully trained {currency_code.upper()} model")
        return True
    except Exception as e:
        print(f"\n✗ Failed to train {currency_code.upper()} model: {e}")
        return False


def train_all_currencies(epochs: int = 100):
    """Train models for all available currencies."""
    with open('api/latest.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    currencies = [item['ab'] for item in data]
    print(f"Found {len(currencies)} currencies to train")

    results = {}
    for currency in currencies:
        success = train_single_currency(currency, epochs)
        results[currency] = success

    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    successful = sum(1 for v in results.values() if v)
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {len(results) - successful}/{len(results)}")

    if successful < len(results):
        print("\nFailed currencies:")
        for currency, success in results.items():
            if not success:
                print(f"  - {currency}")


def main():
    parser = argparse.ArgumentParser(description='Train currency price prediction models')
    parser.add_argument('--currency', type=str, help='Currency code to train (e.g., usd, eur)')
    parser.add_argument('--all', action='store_true', help='Train models for all currencies')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')

    args = parser.parse_args()

    if args.all:
        train_all_currencies(epochs=args.epochs)
    elif args.currency:
        train_single_currency(args.currency, epochs=args.epochs)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
