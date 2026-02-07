# Quick Start Guide

Get your robo-advisor up and running in 10 minutes!

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Gurobi license (optional, can use OR-Tools instead)
- Git

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/robo-advisor.git
cd robo-advisor
```

### 2. Setup Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
nano .env  # or use your favorite editor
```

Required API keys:
- `YAHOO_FINANCE_API_KEY` (optional, yfinance works without key)
- `ALPHA_VANTAGE_API_KEY` (get free key at https://www.alphavantage.co)
- `GUROBI_LICENSE_FILE` (if using Gurobi)

### 4. Start Infrastructure with Docker

```bash
# Start all services (MongoDB, Redis, MLflow, etc.)
docker-compose up -d

# Check services are running
docker-compose ps

# View logs
docker-compose logs -f
```

Services will be available at:
- MongoDB: `localhost:27017`
- Redis: `localhost:6379`
- MLflow: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin/roboadvisor2024)

### 5. Initialize Database

```bash
# Run database initialization script
python scripts/setup_mongodb.py

# Seed with sample data (optional)
python scripts/seed_data.py
```

### 6. Train Initial Models

```bash
# Download market data and train models
python scripts/train_initial_models.py

# This will:
# - Download historical data (5 years)
# - Engineer features
# - Train ensemble models
# - Register models in MLflow
# - Save to model registry
```

### 7. Start Application

#### Option A: Run Locally

```bash
# Terminal 1: Start API
uvicorn src.presentation.api.main:app --reload --port 8000

# Terminal 2: Start Dashboard
streamlit run src.presentation/dashboard/streamlit_app.py

# Terminal 3: Start Celery Worker (background tasks)
celery -A src.infrastructure.tasks worker --loglevel=info

# Terminal 4: Start Celery Beat (scheduler)
celery -A src.infrastructure.tasks beat --loglevel=info
```

#### Option B: Use Docker

```bash
# All services already started in step 4
# API available at: http://localhost:8000
# Dashboard at: http://localhost:8501
```

### 8. Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Run tests
pytest tests/ -v

# Check coverage
pytest --cov=src --cov-report=html
```

## Basic Usage

### CLI Usage

```bash
# Create a portfolio
robo-advisor portfolio create \
  --name "My Portfolio" \
  --capital 100000 \
  --risk-tolerance 0.15

# Optimize portfolio
robo-advisor optimize \
  --portfolio-id <portfolio-id> \
  --strategy markowitz \
  --target-return 0.10

# Calculate risk metrics
robo-advisor risk calculate \
  --portfolio-id <portfolio-id> \
  --confidence 0.95

# Run backtest
robo-advisor backtest \
  --strategy risk-parity \
  --start-date 2020-01-01 \
  --end-date 2024-01-01
```

### Python API Usage

```python
from src.application.use_cases.optimize_portfolio import OptimizePortfolioUseCase
from src.infrastructure.optimization.gurobi_optimizer import GurobiPortfolioOptimizer
from decimal import Decimal
import numpy as np

# Setup
optimizer = GurobiPortfolioOptimizer()
use_case = OptimizePortfolioUseCase(optimizer)

# Sample data
expected_returns = np.array([0.12, 0.10, 0.08, 0.15])
covariance_matrix = np.array([
    [0.04, 0.01, 0.005, 0.02],
    [0.01, 0.03, 0.008, 0.015],
    [0.005, 0.008, 0.02, 0.01],
    [0.02, 0.015, 0.01, 0.05]
])

# Optimize
result = optimizer.optimize_markowitz(
    expected_returns=expected_returns,
    covariance_matrix=covariance_matrix,
    budget=1.0,
    risk_aversion=1.0
)

print(f"Optimal weights: {result[0]}")
print(f"Sharpe ratio: {result[1]['sharpe_ratio']:.4f}")
```

### REST API Usage

```bash
# Create portfolio
curl -X POST "http://localhost:8000/api/v1/portfolios" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Tech Portfolio",
    "initial_capital": 100000,
    "risk_tolerance": 0.15,
    "owner_id": "user123"
  }'

# Optimize portfolio
curl -X POST "http://localhost:8000/api/v1/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio_id": "<portfolio-id>",
    "strategy": "markowitz",
    "target_return": 0.12,
    "constraints": {
      "max_position_size": 0.20,
      "sectors": {
        "tech": [0.0, 0.40],
        "finance": [0.0, 0.30]
      }
    }
  }'

# Get risk metrics
curl "http://localhost:8000/api/v1/portfolios/<portfolio-id>/risk"

# Run stress test
curl -X POST "http://localhost:8000/api/v1/portfolios/<portfolio-id>/stress-test" \
  -H "Content-Type: application/json" \
  -d '{
    "scenario": "2008_financial_crisis"
  }'
```

### Dashboard Usage

1. Navigate to `http://localhost:8501`
2. Select or create a portfolio
3. View current allocations and performance
4. Run optimization with different strategies
5. Analyze risk metrics and stress tests
6. Compare strategies via backtesting
7. Monitor model drift and performance

## Common Tasks

### Update Market Data

```bash
# Manual update
python scripts/update_market_data.py

# Automated (runs daily via Celery Beat)
# Already configured in docker-compose.yml
```

### Retrain Models

```bash
# Automatic retraining when drift detected
# Or manually:
python scripts/retrain_models.py --force
```

### View Logs

```bash
# Application logs
tail -f logs/app.log

# Docker logs
docker-compose logs -f api
docker-compose logs -f celery-worker
```

### Backup Data

```bash
# MongoDB backup
docker exec robo-advisor-mongodb mongodump \
  --out /backup --username admin --password roboadvisor2024

# Copy backup to host
docker cp robo-advisor-mongodb:/backup ./backups/
```

## Troubleshooting

### Problem: Gurobi license error

**Solution**: 
```bash
# Option 1: Use OR-Tools instead (free)
# Set in config/settings.py:
OPTIMIZER_BACKEND = "ortools"

# Option 2: Get Gurobi academic license
# https://www.gurobi.com/academia/academic-program-and-licenses/
export GUROBI_LICENSE_FILE=/path/to/gurobi.lic
```

### Problem: MongoDB connection failed

**Solution**:
```bash
# Check if MongoDB is running
docker-compose ps mongodb

# Restart MongoDB
docker-compose restart mongodb

# Check logs
docker-compose logs mongodb
```

### Problem: Model training fails

**Solution**:
```bash
# Check data availability
python scripts/check_data.py

# Download fresh data
python scripts/download_market_data.py --force

# Train with smaller dataset
python scripts/train_initial_models.py --max-history 1y
```

### Problem: High memory usage

**Solution**:
```bash
# Adjust Docker resources in docker-compose.yml
# Add under services:
    deploy:
      resources:
        limits:
          memory: 4G

# Or reduce batch sizes in config/model_config.yaml
```

## Next Steps

1. **Customize Strategies**: Modify optimization strategies in `src/infrastructure/optimization/`
2. **Add Features**: Engineer new features in `src/infrastructure/ml/feature_engineering/`
3. **Extend API**: Add endpoints in `src/presentation/api/routers/`
4. **Create Dashboards**: Build visualizations in `src/presentation/dashboard/`
5. **Deploy**: See `docs/deployment.md` for production deployment guide

## Getting Help

- Documentation: `docs/`
- API Docs: `http://localhost:8000/docs`
- Issues: GitHub Issues
- Discord: [Community Server]
- Email: support@example.com

## Development

### Run Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# With coverage
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type checking
mypy src/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Generate Documentation

```bash
# API documentation
pdoc --html src/ -o docs/api/

# User documentation
mkdocs serve
# Visit http://localhost:8000
```

## Production Deployment

See `docs/deployment.md` for:
- Kubernetes deployment
- CI/CD pipelines
- Monitoring setup
- Security best practices
- Scaling guidelines

---

**Congratulations! Your robo-advisor is now running! 🎉**
