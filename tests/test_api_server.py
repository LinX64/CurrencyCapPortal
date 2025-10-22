"""
Tests for REST API server.
"""

import pytest
import json
from api_server import app


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test health check returns 200."""
        response = client.get('/health')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data


class TestCurrenciesEndpoint:
    """Test currencies listing endpoint."""

    def test_list_currencies(self, client):
        """Test listing all currencies."""
        response = client.get('/api/v1/currencies')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'currencies' in data
        assert 'count' in data
        assert isinstance(data['currencies'], list)


class TestPredictionsEndpoint:
    """Test predictions endpoints."""

    def test_list_available_predictions(self, client):
        """Test listing available predictions."""
        response = client.get('/api/v1/predictions/available')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'currencies' in data
        assert 'count' in data
        assert isinstance(data['currencies'], list)

    def test_get_predictions_invalid_currency(self, client):
        """Test getting predictions for invalid currency."""
        response = client.get('/api/v1/predictions/invalid_currency')

        data = json.loads(response.data)
        assert 'error' in data

    def test_batch_predictions_no_body(self, client):
        """Test batch predictions without request body."""
        response = client.post('/api/v1/predictions/batch')
        assert response.status_code == 400

        data = json.loads(response.data)
        assert 'error' in data

    def test_batch_predictions_with_empty_list(self, client):
        """Test batch predictions with empty currency list."""
        response = client.post(
            '/api/v1/predictions/batch',
            data=json.dumps({'currencies': [], 'hours': 24}),
            content_type='application/json'
        )
        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'predictions' in data
        assert len(data['predictions']) == 0


class TestModelInfoEndpoint:
    """Test model info endpoint."""

    def test_get_model_info_invalid_currency(self, client):
        """Test getting model info for invalid currency."""
        response = client.get('/api/v1/model/info/invalid_currency')

        data = json.loads(response.data)
        assert 'error' in data


class TestErrorHandlers:
    """Test error handlers."""

    def test_404_handler(self, client):
        """Test 404 error handler."""
        response = client.get('/nonexistent/endpoint')
        assert response.status_code == 404

        data = json.loads(response.data)
        assert 'error' in data
