# AI Price Prediction for Currency Cap Portal

This module adds machine learning-based price prediction capabilities to the Currency Cap Portal using LSTM neural networks.

## Features

- **Time Series Forecasting**: LSTM-based deep learning model for price prediction
- **Feature Engineering**: Automatic creation of technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Multi-horizon Predictions**: Predict prices for next 1-48 hours
- **Automated Training**: Train models for all currencies or specific ones
- **API Integration**: Predictions saved as JSON endpoints alongside historical data

## Installation

Install ML dependencies:

```bash
pip install -r ml_requirements.txt
```

## Quick Start

### 1. Train a Model

Train a model for a specific currency:

```bash
python train_model.py --currency usd --epochs 100
```

Train models for all currencies:

```bash
python train_model.py --all --epochs 100
```

### 2. Generate Predictions

Generate 24-hour predictions for USD:

```bash
python predict_prices.py --currency usd --hours 24
```

Generate predictions for all trained models:

```bash
python predict_prices.py --all --hours 24
```

### 3. Integrate with Updates

Predictions can be automatically generated during data updates by adding to [update_apis.py](update_apis.py):

```python
from ml.updater import update_predictions

async def main():
    # ... existing update code ...

    await update_predictions(prediction_hours=24)
```

## Data Requirements

### Current Data (Available)
- Historical buy/sell prices with timestamps
- Multiple time periods (1d to all-time)
- Updated every 5 minutes via GitHub Actions

### Features Engineered
- Time-based features (hour, day of week, month, quarter)
- Moving averages (7, 30, 90 days)
- Price momentum and rate of change
- Volatility metrics
- Technical indicators (RSI, MACD, Bollinger Bands)
- Spread analysis

### Additional Data That Would Improve Predictions
While the model works with current data, these additions would improve accuracy:

1. **Trading Volume**: Transaction volume data for liquidity indicators
2. **Economic Indicators**:
   - GDP growth rates
   - Inflation rates
   - Interest rates
   - Employment data
3. **Market Sentiment**:
   - News sentiment scores
   - Social media trends
   - Market fear/greed indices
4. **Global Events**: Political events, policy changes, major announcements
5. **Cross-Currency Correlations**: Relationships between different currency pairs
6. **Order Book Data**: Bid/ask spreads and depth

## Model Architecture

### LSTM Network
```
Input (sequence_length, n_features)
    ↓
LSTM Layer (128 units) + Dropout
    ↓
LSTM Layer (64 units) + Dropout
    ↓
LSTM Layer (32 units) + Dropout
    ↓
Dense Layer (16 units, ReLU) + Dropout
    ↓
Output Layer (1 unit) - Predicted Price
```

### Training Configuration
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: MAE, MAPE
- **Callbacks**: Early stopping, learning rate reduction, model checkpointing
- **Default Sequence Length**: 30 time steps

## API Endpoints

After training and prediction, the following endpoints are available:

### Prediction Data
```
/api/predictions/index.json       - Index of all available predictions
/api/predictions/{currency}.json  - Predictions for specific currency
```

### Example Prediction Response
```json
{
  "currency_code": "usd",
  "generated_at": "2025-10-22T20:00:00Z",
  "prediction_horizon_hours": 24,
  "model_info": {
    "trained_at": "2025-10-22T15:30:00Z",
    "test_mae": 45.23,
    "test_mape": 0.042
  },
  "predictions": [
    {
      "timestamp": "2025-10-22T21:00:00Z",
      "predicted_price": 107850.5,
      "hours_ahead": 1
    },
    {
      "timestamp": "2025-10-22T22:00:00Z",
      "predicted_price": 107920.3,
      "hours_ahead": 2
    }
  ]
}
```

## File Structure

```
ml/
├── __init__.py           - Module exports
├── data_processor.py     - Data loading and feature engineering
├── model.py              - LSTM model architecture
├── predictor.py          - High-level prediction interface
└── updater.py            - Integration with update pipeline

models/                   - Trained model storage
├── {currency}_model.keras
├── {currency}_scalers.pkl
└── {currency}_metadata.json

api/predictions/          - Generated predictions
├── index.json
└── {currency}.json
```

## Usage Examples

### Python API

```python
from ml.predictor import CurrencyPredictor

predictor = CurrencyPredictor(currency_code='usd')

predictor.train_model(epochs=100)

predictions = predictor.predict_future(hours=24)

for pred in predictions:
    print(f"{pred['timestamp']}: {pred['predicted_price']:.2f}")
```

### Training Multiple Currencies

```python
currencies = ['usd', 'eur', 'gbp', 'jpy']
for currency in currencies:
    predictor = CurrencyPredictor(currency_code=currency)
    predictor.train_model(epochs=100)
```

## Model Performance

Model performance varies by currency based on:
- Data availability and quality
- Market volatility
- Liquidity and trading volume
- External market factors

Expected metrics for stable currencies:
- **MAE**: 0.1-1% of average price
- **MAPE**: 2-5% for short-term predictions (1-6 hours)

## Limitations

1. **Short-term Focus**: Most accurate for 1-6 hour predictions
2. **Market Events**: Cannot predict impact of unforeseen events
3. **Data Gaps**: Performance degrades with missing or sparse data
4. **No Volume Data**: Cannot account for liquidity changes
5. **Single Asset**: Each currency predicted independently (no cross-correlation)

## Future Enhancements

1. **Ensemble Models**: Combine LSTM with other algorithms (XGBoost, Prophet)
2. **Attention Mechanisms**: Add transformer-based attention layers
3. **Multi-asset Models**: Train on multiple correlated currencies
4. **Uncertainty Quantification**: Provide confidence intervals
5. **Online Learning**: Continuously update models with new data
6. **External Data Integration**: Incorporate news, sentiment, economic indicators
7. **Automated Retraining**: Trigger retraining when performance degrades

## Troubleshooting

### Model Training Fails
- Check data availability: `python -c "from ml.predictor import CurrencyPredictor; p = CurrencyPredictor('usd'); print(len(p.data_processor.load_currency_data('usd')))"`
- Ensure sufficient historical data (minimum 100 data points recommended)
- Check disk space for model storage

### Poor Predictions
- Retrain with more epochs
- Increase sequence length for longer patterns
- Add more historical data
- Check for data quality issues

### Memory Issues
- Reduce batch size in training
- Use smaller sequence length
- Train currencies sequentially instead of in parallel

## Contributing

When adding new features:
1. Maintain backward compatibility with existing models
2. Add tests for new functionality
3. Update model metadata version
4. Document new hyperparameters

## License

Same as Currency Cap Portal main project.
