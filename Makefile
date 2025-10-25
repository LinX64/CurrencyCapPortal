.PHONY: help test test-verbose test-fast install clean lint format security

help:
	@echo "Currency Cap Portal - Development Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make install        - Install all dependencies including test dependencies"
	@echo "  make install-ml     - Install ML dependencies"
	@echo "  make test          - Run all tests"
	@echo "  make test-verbose  - Run tests with verbose output"
	@echo "  make test-fast     - Run tests without slow tests"
	@echo "  make lint          - Run code linting"
	@echo "  make format        - Format code with black and isort"
	@echo "  make security      - Run security checks"
	@echo "  make clean         - Clean up cache and temporary files"
	@echo "  make update-apis   - Update all API data"
	@echo ""
	@echo "ML Commands:"
	@echo "  make train-model    - Train ML model for a currency (CURRENCY=usd)"
	@echo "  make train-all      - Train models for all currencies"
	@echo "  make predict        - Generate predictions (CURRENCY=usd HOURS=24)"
	@echo "  make predict-all    - Generate predictions for all models"
	@echo "  make api-server     - Start REST API server"
	@echo "  make test-ml        - Run ML tests only"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest

test-verbose:
	pytest -v

test-fast:
	pytest -m "not slow"

test-watch:
	pytest-watch

lint:
	@echo "Running flake8..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=tests,venv,.venv,env,.env
	flake8 . --count --max-complexity=10 --max-line-length=120 --statistics --exclude=tests,venv,.venv,env,.env
	@echo "Running black check..."
	black --check --exclude='/(\.git|\.venv|venv|env|\.env|__pycache__|\.pytest_cache)/' .
	@echo "Running isort check..."
	isort --check-only --skip .venv --skip venv --skip env --skip .env .

format:
	@echo "Formatting with black..."
	black --exclude='/(\.git|\.venv|venv|env|\.env|__pycache__|\.pytest_cache)/' .
	@echo "Sorting imports with isort..."
	isort --skip .venv --skip venv --skip env --skip .env .

security:
	@echo "Checking dependencies for security issues..."
	safety check || true
	@echo "Running security analysis with bandit..."
	bandit -r . --exclude './tests,./venv,./.venv,./env,./.env' || true

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache/
	@echo "Clean complete!"

update-apis:
	python update_apis.py

# Run specific test file
test-helper:
	pytest tests/test_helper.py -v

test-cache:
	pytest tests/test_cache.py -v

test-updaters:
	pytest tests/test_updaters.py -v

test-apis:
	pytest tests/test_APIs.py -v

test-generate:
	pytest tests/test_generate_history_date.py -v

test-update-apis:
	pytest tests/test_update_apis.py -v

install-ml:
	pip install -r ml_requirements.txt

train-model:
	@if [ -z "$(CURRENCY)" ]; then \
		echo "Error: CURRENCY not specified. Usage: make train-model CURRENCY=usd"; \
		exit 1; \
	fi
	python train_model.py --currency $(CURRENCY) --epochs $(or $(EPOCHS),100)

train-all:
	python train_model.py --all --epochs $(or $(EPOCHS),100)

predict:
	@if [ -z "$(CURRENCY)" ]; then \
		echo "Error: CURRENCY not specified. Usage: make predict CURRENCY=usd HOURS=24"; \
		exit 1; \
	fi
	python predict_prices.py --currency $(CURRENCY) --hours $(or $(HOURS),24)

predict-all:
	python predict_prices.py --all --hours $(or $(HOURS),24)

api-server:
	python api_server.py

api-server-prod:
	gunicorn -w 4 -b 0.0.0.0:5000 api_server:app

test-ml:
	pytest tests/test_ml_predictor.py tests/test_api_server.py -v
