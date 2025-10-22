"""
LSTM model architecture for price prediction.
"""

import os
from typing import Tuple, Optional, List, Dict
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class LSTMPriceModel:
    """LSTM-based price prediction model."""

    def __init__(
        self,
        sequence_length: int = 30,
        n_features: int = 16,
        lstm_units: List[int] = None,
        dropout_rate: float = 0.2
    ):
        """
        Initialize the LSTM model.

        Args:
            sequence_length: Number of time steps in input sequence
            n_features: Number of features per time step
            lstm_units: List of units for each LSTM layer
            dropout_rate: Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units or [128, 64, 32]
        self.dropout_rate = dropout_rate
        self.model = None

    def build_model(self) -> keras.Model:
        """
        Build the LSTM model architecture.

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential()

        # Input layer
        model.add(layers.Input(shape=(self.sequence_length, self.n_features)))

        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            model.add(layers.LSTM(
                units,
                return_sequences=return_sequences,
                kernel_regularizer=keras.regularizers.l2(0.001)
            ))
            model.add(layers.Dropout(self.dropout_rate))

        # Dense layers
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(1))

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )

        self.model = model
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        model_path: str = 'models/price_predictor.keras'
    ) -> keras.callbacks.History:
        """
        Train the model.

        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            model_path: Path to save best model

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input sequences

        Returns:
            Predicted prices
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")

        return self.model.predict(X, verbose=0)

    def predict_multiple_steps(
        self,
        initial_sequence: np.ndarray,
        n_steps: int = 24
    ) -> np.ndarray:
        """
        Predict multiple steps into the future.

        Args:
            initial_sequence: Initial sequence (1, sequence_length, n_features)
            n_steps: Number of steps to predict

        Returns:
            Array of predictions
        """
        predictions = []
        current_sequence = initial_sequence.copy()

        for _ in range(n_steps):
            # Predict next step
            pred = self.predict(current_sequence)[0, 0]
            predictions.append(pred)

            # Update sequence (simplified - using only predicted price)
            # In practice, you'd need to update all features
            new_step = current_sequence[0, -1, :].copy()
            new_step[0] = pred  # Update avg_price feature

            # Shift sequence and add new prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = new_step

        return np.array(predictions)

    def save(self, path: str):
        """Save the model."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(path)

    def load(self, path: str):
        """Load a trained model."""
        self.model = keras.models.load_model(path)
        return self

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the model.

        Args:
            X_test: Test sequences
            y_test: Test targets

        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")

        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = {
            'loss': results[0],
            'mae': results[1],
            'mape': results[2]
        }

        return metrics
