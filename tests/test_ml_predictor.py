"""
Tests for ML prediction functionality.
"""

import pytest
import os
import json
import numpy as np
from ml.data_processor import DataProcessor
from ml.model import LSTMPriceModel
from ml.predictor import CurrencyPredictor


@pytest.fixture
def sample_currency_data():
    """Create sample currency data for testing."""
    return [
        {
            "ab": "usd",
            "av": "🇺🇸",
            "en": "US Dollar",
            "fa": "دلار آمریکا",
            "ps": [
                {"bp": 107700, "sp": 107810, "ts": "2025-10-22T20:00:00Z"},
                {"bp": 107750, "sp": 107860, "ts": "2025-10-22T21:00:00Z"},
                {"bp": 107800, "sp": 107910, "ts": "2025-10-22T22:00:00Z"}
            ]
        }
    ]


@pytest.fixture
def temp_data_file(tmp_path, sample_currency_data):
    """Create temporary data file for testing."""
    data_file = tmp_path / "test_data.json"
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(sample_currency_data, f)
    return str(data_file)


class TestDataProcessor:
    """Test data processing functionality."""

    def test_load_currency_data(self, temp_data_file):
        """Test loading currency data from JSON."""
        processor = DataProcessor()
        df = processor.load_currency_data('usd', temp_data_file)

        assert len(df) == 3
        assert 'timestamp' in df.columns
        assert 'buy_price' in df.columns
        assert 'sell_price' in df.columns
        assert 'avg_price' in df.columns
        assert 'spread' in df.columns

    def test_load_nonexistent_currency(self, temp_data_file):
        """Test loading non-existent currency raises error."""
        processor = DataProcessor()

        with pytest.raises(ValueError, match="Currency xyz not found"):
            processor.load_currency_data('xyz', temp_data_file)

    def test_engineer_features(self):
        """Test feature engineering."""
        processor = DataProcessor()

        import pandas as pd
        from datetime import datetime, timedelta

        base_time = datetime.now()
        data = {
            'timestamp': [base_time + timedelta(hours=i) for i in range(100)],
            'buy_price': [100 + i for i in range(100)],
            'sell_price': [101 + i for i in range(100)]
        }
        df = pd.DataFrame(data)
        df['avg_price'] = (df['buy_price'] + df['sell_price']) / 2
        df['spread'] = df['sell_price'] - df['buy_price']

        df_features = processor.engineer_features(df)

        assert 'hour' in df_features.columns
        assert 'day_of_week' in df_features.columns
        assert 'ma_7' in df_features.columns
        assert 'ma_30' in df_features.columns
        assert 'volatility_7' in df_features.columns
        assert 'price_change_pct' in df_features.columns

    def test_create_sequences(self):
        """Test sequence creation for LSTM."""
        processor = DataProcessor()

        import pandas as pd
        from datetime import datetime, timedelta

        n_samples = 100
        base_time = datetime.now()
        data = {
            'timestamp': [base_time + timedelta(hours=i) for i in range(n_samples)],
            'avg_price': [100 + i for i in range(n_samples)],
            'spread': [1] * n_samples,
            'spread_pct': [1] * n_samples,
            'hour': [i % 24 for i in range(n_samples)],
            'day_of_week': [i % 7 for i in range(n_samples)],
            'month': [1] * n_samples,
            'price_change_pct': [0.01] * n_samples,
            'ma_7': [100] * n_samples,
            'ma_30': [100] * n_samples,
            'ma_90': [100] * n_samples,
            'volatility_7': [1] * n_samples,
            'volatility_30': [1] * n_samples,
            'momentum_7': [1] * n_samples,
            'momentum_30': [1] * n_samples,
            'roc_7': [1] * n_samples,
            'roc_30': [1] * n_samples
        }
        df = pd.DataFrame(data)

        sequence_length = 30
        X, y, feature_names = processor.create_sequences(df, sequence_length=sequence_length)

        assert X.shape[1] == sequence_length
        assert len(y) == len(X)
        assert len(feature_names) > 0


class TestLSTMPriceModel:
    """Test LSTM model functionality."""

    def test_build_model(self):
        """Test model building."""
        model = LSTMPriceModel(sequence_length=30, n_features=16)
        keras_model = model.build_model()

        assert keras_model is not None
        assert len(keras_model.layers) > 0

    def test_predict_shape(self):
        """Test prediction output shape."""
        model = LSTMPriceModel(sequence_length=30, n_features=16)
        model.build_model()

        X_test = np.random.rand(10, 30, 16)
        predictions = model.predict(X_test)

        assert predictions.shape == (10, 1)


class TestCurrencyPredictor:
    """Test high-level predictor interface."""

    @pytest.fixture
    def predictor(self, tmp_path):
        """Create predictor with temporary model directory."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        return CurrencyPredictor(currency_code='usd', model_dir=str(model_dir))

    def test_predictor_initialization(self, predictor):
        """Test predictor initialization."""
        assert predictor.currency_code == 'usd'
        assert predictor.data_processor is not None
        assert predictor.model is None

    def test_get_model_info_no_model(self, predictor):
        """Test getting model info when no model exists."""
        info = predictor.get_model_info()
        assert info is None


def test_imports():
    """Test that all modules can be imported."""
    from ml import CurrencyPredictor, DataProcessor, LSTMPriceModel

    assert CurrencyPredictor is not None
    assert DataProcessor is not None
    assert LSTMPriceModel is not None
