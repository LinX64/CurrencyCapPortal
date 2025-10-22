#!/usr/bin/env python3
"""
REST API server for currency price predictions.
Mobile apps and web clients can use this API to get predictions.

Usage:
    python api_server.py

    Or for production:
    gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from flask import Flask, jsonify, request
from flask_cors import CORS
from ml.predictor import CurrencyPredictor

app = Flask(__name__)
CORS(app)

MODELS_DIR = 'models'
API_DIR = 'api'


def get_available_currencies() -> List[str]:
    """Get list of currencies with trained models."""
    if not os.path.exists(MODELS_DIR):
        return []

    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('_model.keras')]
    return [f.replace('_model.keras', '') for f in model_files]


def get_all_currencies() -> List[Dict]:
    """Get all available currencies from latest.json."""
    latest_path = os.path.join(API_DIR, 'latest.json')
    if not os.path.exists(latest_path):
        return []

    with open(latest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Currency Price Prediction API'
    })


@app.route('/api/v1/currencies', methods=['GET'])
def list_currencies():
    """
    List all available currencies.

    Response:
        {
            "currencies": [...],
            "count": 10
        }
    """
    currencies = get_all_currencies()
    available_predictions = get_available_currencies()

    for currency in currencies:
        currency['has_prediction_model'] = currency['ab'] in available_predictions

    return jsonify({
        'currencies': currencies,
        'count': len(currencies),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/v1/predictions/available', methods=['GET'])
def list_available_predictions():
    """
    List currencies with trained prediction models.

    Response:
        {
            "currencies": ["usd", "eur", ...],
            "count": 5
        }
    """
    currencies = get_available_currencies()
    return jsonify({
        'currencies': currencies,
        'count': len(currencies),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/v1/predictions/<currency_code>', methods=['GET'])
def get_predictions(currency_code: str):
    """
    Get price predictions for a specific currency.

    Query Parameters:
        hours (int): Prediction horizon in hours (default: 24, max: 168)

    Response:
        {
            "currency_code": "usd",
            "currency_info": {...},
            "current_price": 107850.5,
            "predictions": [...],
            "model_info": {...}
        }
    """
    currency_code = currency_code.lower()

    if currency_code not in get_available_currencies():
        return jsonify({
            'error': 'No trained model found for this currency',
            'currency_code': currency_code,
            'available_currencies': get_available_currencies()
        }), 404

    hours = request.args.get('hours', default=24, type=int)
    hours = min(max(1, hours), 168)

    try:
        predictor = CurrencyPredictor(currency_code=currency_code)
        predictions = predictor.predict_future(hours=hours)
        model_info = predictor.get_model_info()

        currencies = get_all_currencies()
        currency_info = next((c for c in currencies if c['ab'] == currency_code), None)

        current_price = None
        if currency_info and currency_info.get('ps'):
            latest = currency_info['ps'][-1]
            current_price = (latest['bp'] + latest['sp']) / 2

        return jsonify({
            'currency_code': currency_code,
            'currency_info': currency_info,
            'current_price': current_price,
            'prediction_horizon_hours': hours,
            'predictions': predictions,
            'model_info': {
                'trained_at': model_info.get('trained_at'),
                'test_mae': model_info['metrics']['mae'],
                'test_mape': model_info['metrics']['mape'],
                'data_points': model_info.get('data_points')
            },
            'generated_at': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'currency_code': currency_code
        }), 500


@app.route('/api/v1/predictions/<currency_code>/summary', methods=['GET'])
def get_prediction_summary(currency_code: str):
    """
    Get a summary of predictions for a currency.

    Response:
        {
            "currency_code": "usd",
            "current_price": 107850.5,
            "predicted_1h": 107920.3,
            "predicted_6h": 108150.2,
            "predicted_24h": 108500.0,
            "trend": "up",
            "change_24h_percent": 0.6
        }
    """
    currency_code = currency_code.lower()

    if currency_code not in get_available_currencies():
        return jsonify({
            'error': 'No trained model found for this currency',
            'currency_code': currency_code
        }), 404

    try:
        predictor = CurrencyPredictor(currency_code=currency_code)
        predictions = predictor.predict_future(hours=24)

        currencies = get_all_currencies()
        currency_info = next((c for c in currencies if c['ab'] == currency_code), None)

        current_price = None
        if currency_info and currency_info.get('ps'):
            latest = currency_info['ps'][-1]
            current_price = (latest['bp'] + latest['sp']) / 2

        pred_1h = predictions[0]['predicted_price'] if len(predictions) > 0 else None
        pred_6h = predictions[5]['predicted_price'] if len(predictions) > 5 else None
        pred_24h = predictions[23]['predicted_price'] if len(predictions) > 23 else None

        trend = 'neutral'
        change_24h_percent = 0
        if current_price and pred_24h:
            change_24h_percent = ((pred_24h - current_price) / current_price) * 100
            trend = 'up' if change_24h_percent > 0.1 else ('down' if change_24h_percent < -0.1 else 'neutral')

        return jsonify({
            'currency_code': currency_code,
            'currency_info': currency_info,
            'current_price': current_price,
            'predicted_1h': pred_1h,
            'predicted_6h': pred_6h,
            'predicted_24h': pred_24h,
            'trend': trend,
            'change_24h_percent': round(change_24h_percent, 2),
            'generated_at': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'currency_code': currency_code
        }), 500


@app.route('/api/v1/predictions/batch', methods=['POST'])
def get_batch_predictions():
    """
    Get predictions for multiple currencies at once.

    Request Body:
        {
            "currencies": ["usd", "eur", "gbp"],
            "hours": 24
        }

    Response:
        {
            "predictions": {
                "usd": {...},
                "eur": {...}
            },
            "errors": {
                "gbp": "No trained model"
            }
        }
    """
    data = request.get_json()

    if not data or 'currencies' not in data:
        return jsonify({'error': 'Missing currencies in request body'}), 400

    currencies = data.get('currencies', [])
    hours = min(max(1, data.get('hours', 24)), 168)

    results = {}
    errors = {}

    for currency_code in currencies:
        currency_code = currency_code.lower()

        if currency_code not in get_available_currencies():
            errors[currency_code] = 'No trained model found'
            continue

        try:
            predictor = CurrencyPredictor(currency_code=currency_code)
            predictions = predictor.predict_future(hours=hours)
            model_info = predictor.get_model_info()

            results[currency_code] = {
                'predictions': predictions,
                'model_info': {
                    'test_mae': model_info['metrics']['mae'],
                    'test_mape': model_info['metrics']['mape']
                }
            }
        except Exception as e:
            errors[currency_code] = str(e)

    return jsonify({
        'predictions': results,
        'errors': errors,
        'generated_at': datetime.now().isoformat()
    })


@app.route('/api/v1/model/info/<currency_code>', methods=['GET'])
def get_model_info(currency_code: str):
    """
    Get detailed information about a trained model.

    Response:
        {
            "currency_code": "usd",
            "trained_at": "2025-10-22T15:30:00Z",
            "metrics": {...},
            "data_points": 1500,
            "features": [...]
        }
    """
    currency_code = currency_code.lower()

    if currency_code not in get_available_currencies():
        return jsonify({
            'error': 'No trained model found for this currency',
            'currency_code': currency_code
        }), 404

    try:
        predictor = CurrencyPredictor(currency_code=currency_code)
        model_info = predictor.get_model_info()

        return jsonify(model_info)

    except Exception as e:
        return jsonify({
            'error': str(e),
            'currency_code': currency_code
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    print(f"Starting Currency Prediction API on port {port}")
    print(f"Available models: {len(get_available_currencies())}")
    print(f"Debug mode: {debug}")

    app.run(host='0.0.0.0', port=port, debug=debug)
