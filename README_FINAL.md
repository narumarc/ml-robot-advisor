# 🏦 Robo-Advisor Portfolio Optimization Platform

## 📋 Projet Complet - Data Scientist Banking

Plateforme complète d'optimisation de portefeuille avec ML, MLOps, et Architecture Hexagonale.

---

## 🎯 Vue d'Ensemble

Ce projet démontre:
- ✅ **Architecture Hexagonale** (Ports & Adapters) + DDD
- ✅ **Optimisation Mathématique** (Gurobi: Markowitz, Risk Parity, CVaR)
- ✅ **Machine Learning** (XGBoost, LSTM, GARCH pour prédictions)
- ✅ **MLOps Complet** (Training, Monitoring, Retraining, Deployment)
- ✅ **Risk Management** (VaR, ES, Stress Testing, Backtesting)
- ✅ **ETL Pipeline** (Data extraction, feature engineering)
- ✅ **Infrastructure** (MongoDB, Redis, Docker, Airflow)

---

## 📂 Structure du Projet

```
robo-advisor-project/
├── src/
│   ├── domain/                 # Core Business Logic
│   │   ├── entities/          # Portfolio, Asset
│   │   └── ports/             # Interfaces (IPortfolioRepository, etc.)
│   │
│   ├── application/           # Use Cases
│   │   └── use_cases/         # OptimizePortfolio, PredictReturns
│   │
│   ├── infrastructure/        # Adapters
│   │   ├── persistence/       # MongoDB, Redis
│   │   ├── ml/               # ML Models, Training, Monitoring
│   │   ├── risk_management/  # VaR, ES, Stress Testing
│   │   ├── optimization/     # Gurobi Optimizer
│   │   ├── data_sources/     # yFinance, Alpha Vantage
│   │   └── etl/              # ETL Pipeline
│   │
│   └── presentation/          # API, CLI
│       └── api/              # FastAPI REST API
│
├── mlops/                     # MLOps Scripts
│   ├── training/             # train_all_models.py
│   ├── monitoring/           # check_drift.py, check_performance.py
│   ├── retraining/           # auto_retrain_pipeline.py
│   ├── deployment/           # deploy_model.py, rollback_model.py
│   └── airflow/dags/         # Airflow DAGs
│
├── config/                    # Configuration
├── tests/                     # Tests (unit, integration, e2e)
├── docs/                      # Documentation
└── docker/                    # Docker setup
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone ou extraire le projet
cd robo-advisor-project

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer dépendances
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copier config exemple
cp .env.example .env

# Éditer .env avec vos credentials
nano .env
```

### 3. Lancer Infrastructure (Docker)

```bash
# Démarrer MongoDB, Redis, MLflow, etc.
docker-compose up -d

# Vérifier
docker-compose ps
```

### 4. Entraîner les Modèles

```bash
# Entraîner tous les modèles
python mlops/training/train_all_models.py

# Ou individuellement
python mlops/training/train_return_predictor.py
```

### 5. Lancer l'API

```bash
# Démarrer FastAPI
uvicorn src.presentation.api.main:app --reload --port 8000

# Accéder à la doc: http://localhost:8000/docs
```

---

## 📚 Documentation

### Documents Principaux

1. **ARCHITECTURE.md** - Architecture détaillée (Clean Architecture + DDD)
2. **PORTS_AND_ADAPTERS.md** - Explication Ports & Adapters
3. **ML_VS_MLOPS.md** - Distinction ML infrastructure vs MLOps scripts
4. **ML_INFRASTRUCTURE_COMPLETE.md** - Documentation ML complète
5. **MLOPS_COMPLETE.md** - Documentation MLOps complète
6. **TECHNOLOGIES.md** - Liste des technologies utilisées

### Quick References

- **API Usage**: `docs/QUICKSTART.md`
- **Optimization**: Voir `src/infrastructure/optimization/`
- **ML Training**: Voir `src/infrastructure/ml/training/`
- **Risk Metrics**: Voir `src/infrastructure/risk_management/`

---

## 🎯 Fonctionnalités Principales

### 1. Optimisation de Portefeuille

```python
from src.infrastructure.optimization.portfolio_optimizer import GurobiOptimizer

optimizer = GurobiOptimizer()

# Markowitz - Maximize Sharpe
result = optimizer.optimize_markowitz(
    expected_returns=expected_returns,
    cov_matrix=cov_matrix,
    risk_free_rate=0.02
)

# Risk Parity
result = optimizer.optimize_risk_parity(
    cov_matrix=cov_matrix,
    expected_returns=expected_returns
)

# CVaR Optimization
result = optimizer.optimize_cvar(
    returns_scenarios=scenarios,
    alpha=0.95
)
```

### 2. Machine Learning

```python
from src.infrastructure.ml.models.return_predictor import XGBoostReturnPredictor

predictor = XGBoostReturnPredictor(n_estimators=100)
metrics = predictor.train(X_train, y_train)
predictions = predictor.predict(X_test)
```

### 3. Risk Management

```python
from src.infrastructure.risk_management.risk_calculator import RiskCalculator

calc = RiskCalculator(risk_free_rate=0.02)
metrics = calc.calculate_all_metrics(returns)

print(f"VaR 95%: {metrics.var_95:.4f}")
print(f"Expected Shortfall: {metrics.expected_shortfall_95:.4f}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
```

### 4. MLOps Pipeline

```bash
# Check drift
python mlops/monitoring/check_drift.py \
    --reference data/reference.csv \
    --current data/production.csv

# Auto retraining
python mlops/retraining/auto_retrain_pipeline.py

# Deploy
python mlops/deployment/deploy_model.py \
    --model-path models/return_predictor_latest.pkl
```

---

## 🧪 Tests

```bash
# Tous les tests
pytest

# Tests unitaires
pytest tests/unit/

# Tests d'intégration
pytest tests/integration/

# Avec couverture
pytest --cov=src --cov-report=html
```

---

## 📊 Métriques & Monitoring

### MLflow UI

```bash
mlflow ui --port 5000
# Accéder: http://localhost:5000
```

### Prometheus & Grafana

```bash
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

---

## 🔧 Technologies Utilisées

### Core
- **Python 3.10+**
- **Clean Architecture + DDD**
- **Ports & Adapters (Hexagonal)**

### Optimization
- **Gurobi** - Quadratic & Linear Programming
- **CVXPY** - Convex Optimization

### Machine Learning
- **XGBoost** - Gradient Boosting
- **LightGBM** - Fast Gradient Boosting
- **PyTorch** - LSTM models
- **scikit-learn** - ML utilities
- **arch** - GARCH models

### MLOps
- **MLflow** - Experiment tracking
- **Evidently** - Drift detection
- **Airflow** - Workflow orchestration
- **Prometheus** - Metrics
- **Grafana** - Dashboards

### Infrastructure
- **FastAPI** - REST API
- **MongoDB** - Document DB
- **Redis** - Cache & Feature Store
- **Docker** - Containerization

### Financial
- **yfinance** - Market data
- **pandas** - Data manipulation
- **numpy** - Numerical computing

---

## 📈 Use Cases

### Portfolio Manager
1. Upload portfolio composition
2. Get optimization recommendations (Markowitz, Risk Parity)
3. View risk metrics (VaR, Sharpe, Sortino)
4. Execute rebalancing

### Risk Analyst
1. Run stress tests (market crash, sector shocks)
2. Calculate VaR & Expected Shortfall
3. Monitor portfolio drawdown
4. Generate risk reports

### Quant Developer
1. Train ML models for return prediction
2. Backtest strategies
3. Optimize hyperparameters
4. Deploy models to production

### Data Scientist
1. Feature engineering (technical indicators)
2. Model training with MLflow
3. Drift detection
4. Performance monitoring

---

## 🎓 Pour les Entretiens

### Points à Mentionner

1. **Architecture**: "J'ai implémenté Clean Architecture avec Ports & Adapters pour séparer le domaine de l'infrastructure"

2. **Optimization**: "J'utilise Gurobi pour résoudre des problèmes d'optimisation quadratique (Markowitz) et linéaire (CVaR)"

3. **ML Pipeline**: "Pipeline complet avec feature engineering, training, cross-validation, et MLflow tracking"

4. **MLOps**: "Infrastructure MLOps avec drift detection automatique, retraining triggers, et déploiement avec rollback"

5. **Risk Management**: "Implémentation complète de VaR, ES, stress testing, et backtesting"

### Démo en Direct

```bash
# 1. Montrer l'architecture
tree src/ -L 3

# 2. Lancer un entraînement
python mlops/training/train_return_predictor.py

# 3. Checker le drift
python mlops/monitoring/check_drift.py --reference data/ref.csv --current data/curr.csv

# 4. Optimiser un portfolio
python -c "from scripts.demo_complete import main; main()"
```

---

## 📝 License

Ce projet est à usage éducatif pour démonstration de compétences.

---

## 👤 Auteur

Créé pour candidature Data Scientist dans le secteur bancaire.

**Compétences démontrées:**
- Clean Architecture & DDD
- Optimisation mathématique (Gurobi)
- Machine Learning (XGBoost, LSTM, GARCH)
- MLOps (Training, Monitoring, Deployment)
- Risk Management (VaR, Stress Testing)
- Infrastructure (Docker, MongoDB, Redis, Airflow)

---

## 📞 Support

Pour questions sur l'implémentation:
1. Consulter `/docs/` pour documentation détaillée
2. Voir exemples dans `/scripts/`
3. Lire les docstrings dans le code

---

**Bon courage pour ton entretien ! 🚀💼**
