.PHONY: help install test lint format clean docker-up docker-down run-api train-models backtest monitor

help:
	@echo "Robo-Advisor Portfolio Optimization - Makefile Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install        - Install dependencies"
	@echo "  make setup          - Full setup (install + docker + init db)"
	@echo ""
	@echo "Development:"
	@echo "  make run-api        - Start FastAPI server"
	@echo "  make test           - Run all tests"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-integration - Run integration tests"
	@echo "  make lint           - Run linters (flake8, mypy)"
	@echo "  make format         - Format code (black, isort)"
	@echo "  make clean          - Clean cache and temp files"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up      - Start all services (MongoDB, Redis, MLflow, etc.)"
	@echo "  make docker-down    - Stop all services"
	@echo "  make docker-logs    - View container logs"
	@echo ""
	@echo "ML & Optimization:"
	@echo "  make train-models   - Train all ML models"
	@echo "  make optimize       - Run portfolio optimization"
	@echo "  make backtest       - Run backtesting"
	@echo "  make detect-drift   - Check for model drift"
	@echo ""
	@echo "Monitoring:"
	@echo "  make monitor        - Open monitoring dashboards"
	@echo "  make mlflow-ui      - Open MLflow UI"

# Installation
install:
	@echo "📦 Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

setup: install docker-up init-db
	@echo "✅ Full setup complete"

init-db:
	@echo "🗄️ Initializing database..."
	python scripts/init_db.py
	@echo "✅ Database initialized"

# Development
run-api:
	@echo "🚀 Starting FastAPI server..."
	uvicorn src.presentation.api.main:app --reload --host 0.0.0.0 --port 8000

# Testing
test:
	@echo "🧪 Running all tests..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	@echo "🧪 Running unit tests..."
	pytest tests/unit/ -v

test-integration:
	@echo "🧪 Running integration tests..."
	pytest tests/integration/ -v

test-e2e:
	@echo "🧪 Running E2E tests..."
	pytest tests/e2e/ -v

# Code Quality
lint:
	@echo "🔍 Running linters..."
	flake8 src/ tests/
	mypy src/
	@echo "✅ Linting complete"

format:
	@echo "🎨 Formatting code..."
	black src/ tests/ config/
	isort src/ tests/ config/
	@echo "✅ Formatting complete"

clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/
	rm -rf .coverage
	@echo "✅ Cleanup complete"

# Docker
docker-up:
	@echo "🐳 Starting Docker services..."
	docker-compose up -d
	@echo "✅ Services started"
	@echo "   MongoDB: localhost:27017"
	@echo "   Redis: localhost:6379"
	@echo "   MLflow: http://localhost:5000"
	@echo "   Prometheus: http://localhost:9090"
	@echo "   Grafana: http://localhost:3000"

docker-down:
	@echo "🐳 Stopping Docker services..."
	docker-compose down
	@echo "✅ Services stopped"

docker-logs:
	docker-compose logs -f

docker-clean:
	@echo "🐳 Cleaning Docker resources..."
	docker-compose down -v
	docker system prune -f
	@echo "✅ Docker cleaned"

# ML Operations
train-models:
	@echo "🤖 Training ML models..."
	python scripts/train_models.py --model all
	@echo "✅ Training complete"

train-returns:
	@echo "🤖 Training return prediction model..."
	python scripts/train_models.py --model returns
	@echo "✅ Return predictor trained"

train-volatility:
	@echo "🤖 Training volatility prediction model..."
	python scripts/train_models.py --model volatility
	@echo "✅ Volatility predictor trained"

# Portfolio Operations
optimize:
	@echo "📊 Running portfolio optimization..."
	python scripts/optimize_portfolio.py
	@echo "✅ Optimization complete"

optimize-markowitz:
	@echo "📊 Markowitz optimization..."
	python scripts/optimize_portfolio.py --method markowitz

optimize-risk-parity:
	@echo "📊 Risk Parity optimization..."
	python scripts/optimize_portfolio.py --method risk-parity

optimize-cvar:
	@echo "📊 CVaR optimization..."
	python scripts/optimize_portfolio.py --method cvar

rebalance:
	@echo "⚖️ Rebalancing portfolio..."
	python scripts/rebalance.py
	@echo "✅ Rebalancing complete"

# Backtesting
backtest:
	@echo "📈 Running backtest..."
	python scripts/backtest.py --start 2020-01-01 --end 2023-12-31
	@echo "✅ Backtest complete"

backtest-quick:
	@echo "📈 Running quick backtest (1 year)..."
	python scripts/backtest.py --start 2023-01-01 --end 2023-12-31
	@echo "✅ Quick backtest complete"

# Risk Management
calculate-risk:
	@echo "⚠️ Calculating risk metrics..."
	python scripts/calculate_risk.py
	@echo "✅ Risk calculation complete"

stress-test:
	@echo "⚠️ Running stress tests..."
	python scripts/stress_test.py
	@echo "✅ Stress test complete"

# Monitoring
detect-drift:
	@echo "🔍 Detecting model drift..."
	python scripts/detect_drift.py
	@echo "✅ Drift detection complete"

monitor:
	@echo "📊 Opening monitoring dashboards..."
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"
	open http://localhost:3000 || xdg-open http://localhost:3000 || echo "Open http://localhost:3000"

mlflow-ui:
	@echo "📊 Opening MLflow UI..."
	open http://localhost:5000 || xdg-open http://localhost:5000 || echo "Open http://localhost:5000"

# Data Operations
fetch-data:
	@echo "📥 Fetching market data..."
	python scripts/fetch_market_data.py
	@echo "✅ Data fetched"

update-features:
	@echo "🔧 Updating feature store..."
	python scripts/update_features.py
	@echo "✅ Features updated"

# ETL Pipeline
run-etl:
	@echo "🔄 Running ETL pipeline..."
	python scripts/run_etl.py
	@echo "✅ ETL complete"

# Airflow (if installed)
airflow-init:
	@echo "🌪️ Initializing Airflow..."
	airflow db init
	airflow users create \
		--username admin \
		--firstname Admin \
		--lastname User \
		--role Admin \
		--email admin@example.com \
		--password admin
	@echo "✅ Airflow initialized"

airflow-start:
	@echo "🌪️ Starting Airflow..."
	airflow webserver -p 8080 &
	airflow scheduler &
	@echo "✅ Airflow started at http://localhost:8080"

# Documentation
docs:
	@echo "📚 Generating documentation..."
	cd docs && make html
	@echo "✅ Documentation generated at docs/_build/html/index.html"

docs-serve:
	@echo "📚 Serving documentation..."
	cd docs/_build/html && python -m http.server 8001

# Database
db-migrate:
	@echo "🗄️ Running database migrations..."
	python scripts/db_migrate.py
	@echo "✅ Migrations complete"

db-seed:
	@echo "🌱 Seeding database with sample data..."
	python scripts/seed_db.py
	@echo "✅ Database seeded"

db-backup:
	@echo "💾 Backing up database..."
	mongodump --uri="mongodb://localhost:27017" --out=backups/$(shell date +%Y%m%d_%H%M%S)
	@echo "✅ Backup complete"

# Production
deploy:
	@echo "🚀 Deploying to production..."
	@echo "⚠️  Not implemented - define your deployment strategy"

# Quick start for demo
demo: docker-up init-db fetch-data train-models optimize backtest
	@echo "✅ Demo setup complete!"
	@echo ""
	@echo "🎉 You can now:"
	@echo "  1. View MLflow UI: http://localhost:5000"
	@echo "  2. View Grafana: http://localhost:3000"
	@echo "  3. Start API: make run-api"
	@echo "  4. Run backtest: make backtest"
