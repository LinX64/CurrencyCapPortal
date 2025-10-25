# Quick Start Guide: AI Price Prediction

Get up and running with AI price predictions in 5 minutes.

## 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r ml_requirements.txt
```

## 2. Train Your First Model

Train a model for USD currency:

```bash
python train_model.py --currency usd --epochs 50
```

This will:
- Load historical USD data
- Engineer 20+ features (moving averages, RSI, MACD, etc.)
- Train an LSTM neural network
- Save the model to `models/usd_model.keras`

Expected training time: 5-15 minutes depending on data size.

## 3. Generate Predictions

```bash
python predict_prices.py --currency usd --hours 24
```

Output will be saved to: `api/predictions/usd.json`

## 4. Start the API Server

```bash
python api_server.py
```

The API will be available at: `http://localhost:5000`

## 5. Test the API

### Get predictions for USD:
```bash
curl http://localhost:5000/api/v1/predictions/usd?hours=24
```

### Get prediction summary:
```bash
curl http://localhost:5000/api/v1/predictions/usd/summary
```

### Check available currencies:
```bash
curl http://localhost:5000/api/v1/predictions/available
```

## Mobile App Integration

### iOS (Swift)
```swift
let url = URL(string: "http://your-server:5000/api/v1/predictions/usd?hours=24")!
let (data, _) = try await URLSession.shared.data(from: url)
let predictions = try JSONDecoder().decode(PredictionResponse.self, from: data)
```

### Android (Kotlin)
```kotlin
val response = apiService.getPredictions("usd", hours = 24)
```

### React Native
```javascript
const predictions = await fetch('http://your-server:5000/api/v1/predictions/usd?hours=24')
  .then(res => res.json());
```

## Advanced Usage

### Train all currencies
```bash
python train_model.py --all --epochs 100
```

### Generate predictions for all models
```bash
python predict_prices.py --all --hours 48
```

### Production deployment with Gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
```

## Troubleshooting

### "No trained model found"
Train a model first: `python train_model.py --currency usd`

### API returns 500 error
Check that model files exist in `models/` directory and data exists in `api/history/all.json`

### Predictions seem inaccurate
1. Retrain with more epochs: `--epochs 200`
2. Ensure you have sufficient historical data (minimum 100 data points)
3. Check model metrics in the API response

## Next Steps

- Read full documentation: [README_ML.md](README_ML.md)
- API reference: [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
- Integrate with GitHub Actions for automated retraining
- Add authentication to the API for production use

## What Data Would Improve Predictions?

While the current system works with available price history, adding these would significantly improve accuracy:

1. **Trading Volume** - Daily transaction volumes
2. **Economic Indicators** - GDP, inflation, interest rates
3. **News Sentiment** - Automated sentiment analysis of financial news
4. **Market Events** - Calendar of known events (elections, policy changes)
5. **Cross-Currency Correlations** - Relationships between pairs

See [README_ML.md](README_ML.md) for detailed information on additional data sources.
