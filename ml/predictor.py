"""
High-level predictor interface for currency price prediction.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import joblib

from .data_processor import DataProcessor
from .model import LSTMPriceModel


class CurrencyPredictor:
    """
    Main interface for training models and making predictions.

    Usage:
        predictor = CurrencyPredictor(currency_code='usd')
        predictor.train_model()
        predictions = predictor.predict_future(hours=24)
    """

    def __init__(
        self,
        currency_code: str,
        sequence_length: int = 30,
        model_dir: str = 'models'
    ):
        self.currency_code = currency_code.lower()
        self.sequence_length = sequence_length
        self.model_dir = model_dir

        self.data_processor = DataProcessor(lookback_days=sequence_length)
        self.model = None

        os.makedirs(model_dir, exist_ok=True)

    def train_model(
        self,
        data_path: str = 'api/history/all.json',
        validation_split: float = 0.2,
        test_split: float = 0.1,
        epochs: int = 100,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Train a new prediction model on historical data.
        """
        print(f"Loading data for {self.currency_code}...")
        df = self.data_processor.load_currency_data(self.currency_code, data_path)

        print(f"Engineering features from {len(df)} data points...")
        df = self.data_processor.engineer_features(df)

        print("Creating sequences...")
        X, y, feature_names = self.data_processor.create_sequences(
            df,
            sequence_length=self.sequence_length,
            prediction_horizon=1
        )

        print(f"Created {len(X)} sequences with {X.shape[2]} features")

        n_samples = len(X)
        test_size = int(n_samples * test_split)
        val_size = int(n_samples * validation_split)
        train_size = n_samples - test_size - val_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]

        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        self.model = LSTMPriceModel(
            sequence_length=self.sequence_length,
            n_features=X.shape[2]
        )

        model_path = os.path.join(self.model_dir, f'{self.currency_code}_model.keras')

        print("Training model...")
        self.model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            model_path=model_path
        )

        print("Evaluating on test set...")
        metrics = self.model.evaluate(X_test, y_test)

        scaler_path = os.path.join(self.model_dir, f'{self.currency_code}_scalers.pkl')
        joblib.dump({
            'price_scaler': self.data_processor.price_scaler,
            'feature_scaler': self.data_processor.feature_scaler
        }, scaler_path)

        metadata = {
            'currency_code': self.currency_code,
            'sequence_length': self.sequence_length,
            'n_features': X.shape[2],
            'feature_names': feature_names,
            'trained_at': datetime.now().isoformat(),
            'data_points': len(df),
            'metrics': metrics
        }

        metadata_path = os.path.join(self.model_dir, f'{self.currency_code}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nTraining complete!")
        print(f"Test MAE: {metrics['mae']:.4f}")
        print(f"Test MAPE: {metrics['mape']:.2f}%")

        return metrics

    def load_model(self) -> bool:
        """Load a previously trained model."""
        model_path = os.path.join(self.model_dir, f'{self.currency_code}_model.keras')
        scaler_path = os.path.join(self.model_dir, f'{self.currency_code}_scalers.pkl')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return False

        scalers = joblib.load(scaler_path)
        self.data_processor.price_scaler = scalers['price_scaler']
        self.data_processor.feature_scaler = scalers['feature_scaler']

        metadata_path = os.path.join(self.model_dir, f'{self.currency_code}_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.model = LSTMPriceModel(
            sequence_length=self.sequence_length,
            n_features=metadata['n_features']
        )
        self.model.load(model_path)

        return True

    def predict_future(
        self,
        hours: int = 24,
        data_path: str = 'api/history/all.json'
    ) -> List[Dict]:
        """
        Predict future prices for the next N hours.
        """
        if self.model is None:
            if not self.load_model():
                raise ValueError("No trained model found. Train a model first.")

        df = self.data_processor.load_currency_data(self.currency_code, data_path)
        df = self.data_processor.engineer_features(df)

        recent_sequence = self.data_processor.prepare_prediction_data(
            df,
            sequence_length=self.sequence_length
        )

        predictions_scaled = self.model.predict_multiple_steps(
            recent_sequence,
            n_steps=hours
        )

        predictions = []
        last_timestamp = df['timestamp'].iloc[-1]

        for i, pred_scaled in enumerate(predictions_scaled):
            pred_price = self.data_processor.inverse_transform_price(
                np.array([[pred_scaled]])
            )

            prediction_time = last_timestamp + timedelta(hours=i+1)

            predictions.append({
                'timestamp': prediction_time.isoformat(),
                'predicted_price': float(pred_price),
                'hours_ahead': i + 1
            })

        return predictions

    def get_model_info(self) -> Optional[Dict]:
        """Get metadata about the trained model."""
        metadata_path = os.path.join(self.model_dir, f'{self.currency_code}_metadata.json')

        if not os.path.exists(metadata_path):
            return None

        with open(metadata_path, 'r') as f:
            return json.load(f)
