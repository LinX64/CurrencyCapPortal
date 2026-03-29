# CurrencyCapPortal — Claude Project Guide

## Project Overview
**Gheymat** — AI-powered Iranian exchange rate prediction platform.
- Flask API server with advanced ML prediction engine (LSTM + ensemble)
- Static JSON data files updated via GitHub Actions (every 5 min)
- Frontend: vanilla JS + HTML/CSS, bilingual (EN/FA), deployed on GitHub Pages
- Auth: JWT + SQLite, free/premium user tiers

## Quick Commands

```bash
# Run development server
python api_server.py

# Run all tests
pytest tests/ -v

# Update currency data manually
python update_apis.py

# Generate AI predictions
python generate_predictions.py

# Run with gunicorn (production)
gunicorn --bind 0.0.0.0:8080 api_server:app
```

## Architecture

```
CurrencyCapPortal/
├── api_server.py          # Flask app — all routes + prediction engine
├── auth.py                # JWT auth module (register/login/middleware)
├── generate_predictions.py # Hourly batch prediction generator
├── update_apis.py         # Data updater (runs every 5 min via CI)
├── enhanced_sentiment.py  # News sentiment analysis
├── api/
│   ├── latest.json        # Current prices (42 currencies)
│   ├── history/           # Historical data: 1d, 1w, 1m, 1y, 5y, all
│   ├── predictions/       # Pre-generated: short, medium, long
│   └── news.json          # Financial news
├── website/               # Frontend (deployed to GitHub Pages)
│   ├── index.html
│   ├── script.js
│   └── styles.css
└── data/
    └── users.db           # SQLite user database (gitignored)
```

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/latest` | — | Current exchange rates |
| GET | `/api/history/<period>` | — | Historical data (1d/1w/1m/1y/5y/all) |
| GET | `/api/predictions/<term>` | — | Pre-generated predictions |
| GET | `/api/news` | — | Financial news |
| GET | `/api/v1/currencies` | — | All currencies with prices |
| GET | `/api/v1/sentiment` | — | Market sentiment |
| POST | `/api/v1/predict` | Premium | Live AI prediction |
| POST | `/api/auth/register` | — | Register user |
| POST | `/api/auth/login` | — | Login, get JWT |
| GET | `/api/auth/me` | Auth | Current user info |
| POST | `/api/auth/refresh` | Auth | Refresh JWT token |

## User Tiers
- **Free**: Real-time prices, historical data, news, pre-generated predictions
- **Premium**: Live AI predictions via `/api/v1/predict`, advanced confidence scores

## Key Environment Variables
```
JWT_SECRET=<strong-random-secret>
NEWS_API_KEY=<newsapi.org-key>
PORT=8080
```

## Data Format
Currency object in `latest.json`:
```json
{"ab": "usd", "av": "🇺🇸", "en": "US Dollar", "fa": "دلار آمریکا", "ty": "cu",
 "ps": [{"bp": 154850, "sp": 154950, "ts": "2026-03-29T20:14:06Z"}]}
```

## Auth Flow
1. POST `/api/auth/register` → `{email, password, name}`
2. POST `/api/auth/login` → returns `{access_token, refresh_token, user}`
3. Add `Authorization: Bearer <access_token>` header for protected routes
4. POST `/api/auth/refresh` with refresh token to get new access token

## Frontend API URL
The frontend fetches prices from GitHub Pages:
```javascript
const API_URL = 'https://linx64.github.io/CurrencyCapPortal/latest.json';
```
For the predict endpoint it uses the Flask server URL stored in `AUTH_API_URL`.

## Development Notes
- ML libraries (TF, XGBoost, etc.) are loaded lazily on first prediction request
- SQLite DB is created automatically at `data/users.db` on first auth request
- `.env` file for local secrets (gitignored)
- Tests live in `tests/` — run with `pytest`
