# Currency Prediction API Documentation

REST API for accessing AI-powered currency price predictions. Built for mobile and web applications.

## Base URL

```
http://your-server:5000
```

## Authentication

Currently no authentication required. Consider adding API keys for production.

## Endpoints

### 1. Health Check

Check if the API is running.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-22T20:00:00Z",
  "service": "Currency Price Prediction API"
}
```

---

### 2. List All Currencies

Get all available currencies with their current prices and model availability.

**Endpoint:** `GET /api/v1/currencies`

**Response:**
```json
{
  "currencies": [
    {
      "ab": "usd",
      "av": "🇺🇸",
      "en": "US Dollar",
      "fa": "دلار آمریکا",
      "ps": [...],
      "has_prediction_model": true
    }
  ],
  "count": 10,
  "timestamp": "2025-10-22T20:00:00Z"
}
```

---

### 3. List Available Predictions

Get currencies that have trained prediction models.

**Endpoint:** `GET /api/v1/predictions/available`

**Response:**
```json
{
  "currencies": ["usd", "eur", "gbp", "jpy"],
  "count": 4,
  "timestamp": "2025-10-22T20:00:00Z"
}
```

---

### 4. Get Predictions

Get detailed price predictions for a specific currency.

**Endpoint:** `GET /api/v1/predictions/{currency_code}`

**Query Parameters:**
- `hours` (optional, int): Prediction horizon in hours. Default: 24, Max: 168

**Example Request:**
```
GET /api/v1/predictions/usd?hours=24
```

**Response:**
```json
{
  "currency_code": "usd",
  "currency_info": {
    "ab": "usd",
    "av": "🇺🇸",
    "en": "US Dollar",
    "fa": "دلار آمریکا"
  },
  "current_price": 107850.5,
  "prediction_horizon_hours": 24,
  "predictions": [
    {
      "timestamp": "2025-10-22T21:00:00Z",
      "predicted_price": 107920.3,
      "hours_ahead": 1
    },
    {
      "timestamp": "2025-10-22T22:00:00Z",
      "predicted_price": 107985.7,
      "hours_ahead": 2
    }
  ],
  "model_info": {
    "trained_at": "2025-10-22T15:30:00Z",
    "test_mae": 45.23,
    "test_mape": 0.042,
    "data_points": 1500
  },
  "generated_at": "2025-10-22T20:00:00Z"
}
```

**Error Response (404):**
```json
{
  "error": "No trained model found for this currency",
  "currency_code": "xyz",
  "available_currencies": ["usd", "eur", "gbp"]
}
```

---

### 5. Get Prediction Summary

Get a quick summary of predictions for a currency.

**Endpoint:** `GET /api/v1/predictions/{currency_code}/summary`

**Example Request:**
```
GET /api/v1/predictions/usd/summary
```

**Response:**
```json
{
  "currency_code": "usd",
  "currency_info": {...},
  "current_price": 107850.5,
  "predicted_1h": 107920.3,
  "predicted_6h": 108150.2,
  "predicted_24h": 108500.0,
  "trend": "up",
  "change_24h_percent": 0.6,
  "generated_at": "2025-10-22T20:00:00Z"
}
```

**Trend Values:**
- `up`: Price expected to increase > 0.1%
- `down`: Price expected to decrease > 0.1%
- `neutral`: Expected change < 0.1%

---

### 6. Batch Predictions

Get predictions for multiple currencies in a single request.

**Endpoint:** `POST /api/v1/predictions/batch`

**Request Body:**
```json
{
  "currencies": ["usd", "eur", "gbp"],
  "hours": 24
}
```

**Response:**
```json
{
  "predictions": {
    "usd": {
      "predictions": [...],
      "model_info": {
        "test_mae": 45.23,
        "test_mape": 0.042
      }
    },
    "eur": {
      "predictions": [...],
      "model_info": {...}
    }
  },
  "errors": {
    "gbp": "No trained model found"
  },
  "generated_at": "2025-10-22T20:00:00Z"
}
```

---

### 7. Get Model Information

Get detailed information about a trained prediction model.

**Endpoint:** `GET /api/v1/model/info/{currency_code}`

**Example Request:**
```
GET /api/v1/model/info/usd
```

**Response:**
```json
{
  "currency_code": "usd",
  "sequence_length": 30,
  "n_features": 24,
  "feature_names": [
    "avg_price",
    "spread",
    "hour",
    "day_of_week",
    "ma_7",
    "ma_30",
    "rsi",
    "macd"
  ],
  "trained_at": "2025-10-22T15:30:00Z",
  "data_points": 1500,
  "metrics": {
    "loss": 0.0012,
    "mae": 45.23,
    "mape": 0.042
  }
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 200  | Success |
| 400  | Bad Request - Invalid parameters |
| 404  | Not Found - Currency or model not found |
| 500  | Internal Server Error |

## Mobile App Integration Examples

### Swift (iOS)

```swift
func getPredictions(for currency: String, hours: Int = 24) async throws -> PredictionResponse {
    let url = URL(string: "http://your-server:5000/api/v1/predictions/\(currency)?hours=\(hours)")!
    let (data, _) = try await URLSession.shared.data(from: url)
    return try JSONDecoder().decode(PredictionResponse.self, from: data)
}

struct PredictionResponse: Codable {
    let currencyCode: String
    let currentPrice: Double
    let predictions: [Prediction]
    let modelInfo: ModelInfo

    enum CodingKeys: String, CodingKey {
        case currencyCode = "currency_code"
        case currentPrice = "current_price"
        case predictions
        case modelInfo = "model_info"
    }
}

struct Prediction: Codable {
    let timestamp: String
    let predictedPrice: Double
    let hoursAhead: Int

    enum CodingKeys: String, CodingKey {
        case timestamp
        case predictedPrice = "predicted_price"
        case hoursAhead = "hours_ahead"
    }
}
```

### Kotlin (Android)

```kotlin
data class PredictionResponse(
    @SerializedName("currency_code") val currencyCode: String,
    @SerializedName("current_price") val currentPrice: Double,
    val predictions: List<Prediction>,
    @SerializedName("model_info") val modelInfo: ModelInfo
)

data class Prediction(
    val timestamp: String,
    @SerializedName("predicted_price") val predictedPrice: Double,
    @SerializedName("hours_ahead") val hoursAhead: Int
)

suspend fun getPredictions(currency: String, hours: Int = 24): PredictionResponse {
    val response = apiService.getPredictions(currency, hours)
    return response
}

interface ApiService {
    @GET("api/v1/predictions/{currency}")
    suspend fun getPredictions(
        @Path("currency") currency: String,
        @Query("hours") hours: Int
    ): PredictionResponse
}
```

### React Native (JavaScript)

```javascript
async function getPredictions(currency, hours = 24) {
  const response = await fetch(
    `http://your-server:5000/api/v1/predictions/${currency}?hours=${hours}`
  );

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

async function getPredictionSummary(currency) {
  const response = await fetch(
    `http://your-server:5000/api/v1/predictions/${currency}/summary`
  );
  return await response.json();
}

async function getBatchPredictions(currencies, hours = 24) {
  const response = await fetch(
    'http://your-server:5000/api/v1/predictions/batch',
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        currencies,
        hours
      })
    }
  );
  return await response.json();
}
```

### Flutter (Dart)

```dart
class PredictionService {
  final String baseUrl = 'http://your-server:5000';

  Future<PredictionResponse> getPredictions(String currency, {int hours = 24}) async {
    final response = await http.get(
      Uri.parse('$baseUrl/api/v1/predictions/$currency?hours=$hours'),
    );

    if (response.statusCode == 200) {
      return PredictionResponse.fromJson(jsonDecode(response.body));
    } else {
      throw Exception('Failed to load predictions');
    }
  }

  Future<PredictionSummary> getSummary(String currency) async {
    final response = await http.get(
      Uri.parse('$baseUrl/api/v1/predictions/$currency/summary'),
    );

    if (response.statusCode == 200) {
      return PredictionSummary.fromJson(jsonDecode(response.body));
    } else {
      throw Exception('Failed to load summary');
    }
  }
}
```

## Deployment

### Local Development

```bash
python api_server.py
```

### Production with Gunicorn

```bash
gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
```

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt -r ml_requirements.txt

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "api_server:app"]
```

### Environment Variables

```bash
export PORT=5000
export DEBUG=False
```

## Rate Limiting

Consider adding rate limiting in production:

```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/api/v1/predictions/<currency_code>')
@limiter.limit("10 per minute")
def get_predictions(currency_code):
    ...
```

## CORS Configuration

CORS is enabled for all origins by default. For production, restrict to your domains:

```python
CORS(app, origins=["https://yourdomain.com", "https://app.yourdomain.com"])
```

## Monitoring

Add monitoring endpoints:

```python
@app.route('/metrics')
def metrics():
    return {
        'total_models': len(get_available_currencies()),
        'uptime': ...,
        'requests_processed': ...
    }
```
