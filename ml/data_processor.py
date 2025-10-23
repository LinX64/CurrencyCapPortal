"""
Data processing and feature engineering for price prediction.
"""

import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import ta


class DataProcessor:
    """Process currency data for machine learning models."""

    def __init__(self, lookback_days: int = 30):
        """
        Initialize the data processor.

        Args:
            lookback_days: Number of days to look back for features
        """
        self.lookback_days = lookback_days
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()

    def load_currency_data(self, currency_code: str, data_path: str = 'api/history/all.json') -> pd.DataFrame:
        """
        Load currency data from JSON file.

        Args:
            currency_code: Currency code (e.g., 'usd', 'eur')
            data_path: Path to historical data file

        Returns:
            DataFrame with timestamp, buy_price, sell_price columns
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Find the currency
        currency_data = None
        for item in data:
            if item['ab'].lower() == currency_code.lower():
                currency_data = item
                break

        if not currency_data:
            raise ValueError(f"Currency {currency_code} not found in data")

        # Extract price history
        records = []
        for price_point in currency_data['ps']:
            # Handle missing buy price in older data
            if 'bp' in price_point:
                buy_price = float(price_point['bp'])
                sell_price = float(price_point['sp'])
            else:
                # Estimate buy price from sell price using typical spread (0.2%)
                sell_price = float(price_point['sp'])
                buy_price = sell_price * 0.998  # Assume 0.2% spread

            records.append({
                'timestamp': pd.to_datetime(price_point['ts']),
                'buy_price': buy_price,
                'sell_price': sell_price
            })

        df = pd.DataFrame(records)
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Calculate average price
        df['avg_price'] = (df['buy_price'] + df['sell_price']) / 2
        df['spread'] = df['sell_price'] - df['buy_price']

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from raw price data.

        Args:
            df: DataFrame with timestamp and price columns

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter

        # Price change features
        df['price_change'] = df['avg_price'].diff()
        df['price_change_pct'] = df['avg_price'].pct_change()

        # Moving averages
        df['ma_7'] = df['avg_price'].rolling(window=7, min_periods=1).mean()
        df['ma_30'] = df['avg_price'].rolling(window=30, min_periods=1).mean()
        df['ma_90'] = df['avg_price'].rolling(window=90, min_periods=1).mean()

        # Volatility features
        df['volatility_7'] = df['avg_price'].rolling(window=7, min_periods=1).std()
        df['volatility_30'] = df['avg_price'].rolling(window=30, min_periods=1).std()

        # Price momentum
        df['momentum_7'] = df['avg_price'] - df['avg_price'].shift(7)
        df['momentum_30'] = df['avg_price'] - df['avg_price'].shift(30)

        # Rate of change
        df['roc_7'] = ((df['avg_price'] - df['avg_price'].shift(7)) / df['avg_price'].shift(7)) * 100
        df['roc_30'] = ((df['avg_price'] - df['avg_price'].shift(30)) / df['avg_price'].shift(30)) * 100

        # Technical indicators if enough data
        if len(df) >= 14:
            # RSI (Relative Strength Index)
            df['rsi'] = ta.momentum.RSIIndicator(df['avg_price'], window=14).rsi()

        if len(df) >= 26:
            # MACD
            macd = ta.trend.MACD(df['avg_price'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()

        # Bollinger Bands
        if len(df) >= 20:
            bollinger = ta.volatility.BollingerBands(df['avg_price'], window=20)
            df['bb_high'] = bollinger.bollinger_hband()
            df['bb_low'] = bollinger.bollinger_lband()
            df['bb_mid'] = bollinger.bollinger_mavg()
            df['bb_width'] = df['bb_high'] - df['bb_low']

        # Spread features
        df['spread_pct'] = (df['spread'] / df['avg_price']) * 100
        df['spread_ma_7'] = df['spread'].rolling(window=7, min_periods=1).mean()

        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)

        return df

    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 30,
        prediction_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create sequences for LSTM training.

        Args:
            df: DataFrame with features
            sequence_length: Number of time steps to look back
            prediction_horizon: Number of steps ahead to predict

        Returns:
            X (sequences), y (targets), feature_names
        """
        # Select features for training
        feature_cols = [
            'avg_price', 'spread', 'spread_pct',
            'hour', 'day_of_week', 'month',
            'price_change_pct',
            'ma_7', 'ma_30', 'ma_90',
            'volatility_7', 'volatility_30',
            'momentum_7', 'momentum_30',
            'roc_7', 'roc_30'
        ]

        # Add technical indicators if available
        if 'rsi' in df.columns:
            feature_cols.append('rsi')
        if 'macd' in df.columns:
            feature_cols.extend(['macd', 'macd_signal', 'macd_diff'])
        if 'bb_width' in df.columns:
            feature_cols.extend(['bb_high', 'bb_low', 'bb_mid', 'bb_width'])

        # Filter to available columns
        feature_cols = [col for col in feature_cols if col in df.columns]

        # Extract features
        features = df[feature_cols].values
        target = df['avg_price'].values

        # Normalize features
        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.price_scaler.fit_transform(target.reshape(-1, 1))

        # Create sequences
        X, y = [], []
        for i in range(len(features_scaled) - sequence_length - prediction_horizon + 1):
            X.append(features_scaled[i:i + sequence_length])
            y.append(target_scaled[i + sequence_length + prediction_horizon - 1])

        return np.array(X), np.array(y), feature_cols

    def prepare_prediction_data(
        self,
        df: pd.DataFrame,
        sequence_length: int = 30
    ) -> np.ndarray:
        """
        Prepare the most recent data for prediction.

        Args:
            df: DataFrame with features
            sequence_length: Number of time steps to look back

        Returns:
            Scaled sequence ready for prediction
        """
        # Select features (same as training)
        feature_cols = [
            'avg_price', 'spread', 'spread_pct',
            'hour', 'day_of_week', 'month',
            'price_change_pct',
            'ma_7', 'ma_30', 'ma_90',
            'volatility_7', 'volatility_30',
            'momentum_7', 'momentum_30',
            'roc_7', 'roc_30'
        ]

        # Add technical indicators if available
        if 'rsi' in df.columns:
            feature_cols.append('rsi')
        if 'macd' in df.columns:
            feature_cols.extend(['macd', 'macd_signal', 'macd_diff'])
        if 'bb_width' in df.columns:
            feature_cols.extend(['bb_high', 'bb_low', 'bb_mid', 'bb_width'])

        # Filter to available columns
        feature_cols = [col for col in feature_cols if col in df.columns]

        # Get last sequence_length rows
        recent_data = df[feature_cols].tail(sequence_length).values

        # Scale
        recent_scaled = self.feature_scaler.transform(recent_data)

        return recent_scaled.reshape(1, sequence_length, -1)

    def inverse_transform_price(self, scaled_price: np.ndarray) -> float:
        """
        Convert scaled price back to original scale.

        Args:
            scaled_price: Scaled price value

        Returns:
            Original price value
        """
        return float(self.price_scaler.inverse_transform(scaled_price.reshape(-1, 1))[0, 0])
