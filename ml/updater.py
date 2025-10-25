"""
Updater for ML predictions - integrates with existing update pipeline.
"""

import os
import asyncio
from typing import Optional, List
from .predictor import CurrencyPredictor


async def update_predictions(prediction_hours: int = 24) -> bool:
    """
    Generate predictions for all trained models and save to api/predictions/.
    This function is designed to be called from update_apis.py.
    """
    print("\n🤖 Updating ML predictions...")

    model_dir = 'models'
    if not os.path.exists(model_dir):
        print("   ⚠ No models directory found, skipping predictions")
        return False

    model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.keras')]
    if not model_files:
        print("   ⚠ No trained models found, skipping predictions")
        return False

    currencies = [f.replace('_model.keras', '') for f in model_files]
    print(f"   Found {len(currencies)} trained models")

    os.makedirs('api/predictions', exist_ok=True)

    successful = 0
    failed = []

    for currency in currencies:
        try:
            predictor = CurrencyPredictor(currency_code=currency)
            predictions = predictor.predict_future(hours=prediction_hours)

            model_info = predictor.get_model_info()
            output = {
                'currency_code': currency,
                'prediction_horizon_hours': prediction_hours,
                'model_info': {
                    'trained_at': model_info.get('trained_at'),
                    'test_mae': model_info['metrics']['mae'],
                    'test_mape': model_info['metrics']['mape']
                },
                'predictions': predictions
            }

            import json
            from datetime import datetime
            output['generated_at'] = datetime.now().isoformat()

            output_path = f'api/predictions/{currency}.json'
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            successful += 1
            print(f"   ✓ {currency.upper()}")

        except Exception as e:
            failed.append(currency)
            print(f"   ✗ {currency.upper()}: {e}")

    import json
    from datetime import datetime
    index_path = 'api/predictions/index.json'
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'currencies': [c for c in currencies if c not in failed],
            'prediction_horizon_hours': prediction_hours,
            'successful': successful,
            'failed': len(failed)
        }, f, indent=2)

    print(f"   Summary: {successful} successful, {len(failed)} failed")
    return successful > 0
