# Robo-Advisor Portfolio Optimization Platform

## 🎯 Vue d'ensemble

Plateforme complète de gestion de portefeuille avec ML, optimisation mathématique, et MLOps pour la finance quantitative.

### Fonctionnalités principales

✅ **Optimisation de Portefeuille (Gurobi/OR-Tools)**
- Optimisation de Markowitz (maximisation du ratio de Sharpe)
- Risk Parity (égalisation des contributions au risque)
- CVaR Optimization (minimisation des pertes extrêmes)
- Black-Litterman avec vues subjectives
- Contraintes cardinals et sectorielles

✅ **Machine Learning & Deep Learning**
- Prédiction des rendements (XGBoost, LightGBM, Random Forest)
- LSTM/Transformer pour séries temporelles
- Prédiction de la volatilité (GARCH, ML)
- Détection d'anomalies de marché

✅ **Gestion des Risques**
- VaR (Value at Risk) et Expected Shortfall
- Stress testing de portefeuille
- Monitoring de la volatilité en temps réel
- Limites de position et d'exposition sectorielle

✅ **MLOps & Monitoring**
- MLflow pour tracking des expériences
- Evidently pour détection de drift
- Retraining automatique
- A/B testing de stratégies
- Prometheus + Grafana pour monitoring

✅ **ETL & Data Pipeline**
- Ingestion de données de marché (yfinance, Alpha Vantage)
- Feature store avec Redis
- MongoDB pour persistance
- Airflow pour orchestration

✅ **Clean Architecture & DDD**
- Domain Layer (Entities, Value Objects, Aggregates)
- Application Layer (Use Cases, DTOs)
- Infrastructure Layer (Repositories, External APIs)
- Tests unitaires, intégration, E2E

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                  │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────┐
│              Application Layer (Use Cases)              │
│  - OptimizePortfolio  - PredictReturns                  │
│  - RebalancePortfolio - CalculateRisk                   │
│  - BacktestStrategy   - DetectAnomalies                 │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────┐
│                   Domain Layer (DDD)                    │
│  Entities: Portfolio, Asset, Transaction                │
│  Aggregates: PortfolioAggregate                         │
│  Services: PortfolioDomainService                       │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────┐
│              Infrastructure Layer                        │
│  - MongoDB (Portfolios, Assets)                         │
│  - Redis (Feature Store, Cache)                         │
│  - Gurobi (Optimization)                                │
│  - MLflow (Model Registry)                              │
│  - Airflow (Orchestration)                              │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prérequis

- Python 3.10+
- Docker & Docker Compose
- Gurobi license (ou utiliser OR-Tools/CVXPY)
- MongoDB, Redis

### Installation rapide

```bash
# Cloner le projet
git clone <repo-url>
cd robo-advisor-project

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer dépendances
pip install -r requirements.txt

# Configuration
cp .env.example .env
# Éditer .env avec vos clés API et configurations

# Lancer l'infrastructure
docker-compose up -d

# Initialiser la base de données
python scripts/init_db.py

# Lancer les tests
pytest tests/

# Démarrer l'API
uvicorn src.presentation.api.main:app --reload
```

## 📊 Utilisation

### 1. Optimisation de Portefeuille

```python
from src.application.use_cases.optimize_portfolio import OptimizePortfolioUseCase
from src.infrastructure.optimization.portfolio_optimizer import PortfolioOptimizer

# Préparer les données
expected_returns = pd.Series({
    'AAPL': 0.12,
    'MSFT': 0.10,
    'GOOGL': 0.15,
    'AMZN': 0.13
})

covariance_matrix = pd.DataFrame(...)  # Matrice de covariance

# Optimiser avec Markowitz
optimizer = PortfolioOptimizer()
result = optimizer.optimize_markowitz(
    expected_returns=expected_returns,
    covariance_matrix=covariance_matrix,
    risk_free_rate=0.02,
    max_position_size=0.15
)

print(f"Poids optimaux: {result.weights}")
print(f"Ratio de Sharpe: {result.sharpe_ratio:.4f}")
```

### 2. Prédiction ML des Rendements

```python
from src.infrastructure.ml.models.return_predictor import ReturnPredictor

# Préparer les features
predictor = ReturnPredictor(model_type='xgboost')
features = predictor.prepare_features(prices_df)

# Entraîner le modèle
metrics = predictor.train(
    X=features,
    y=future_returns,
    validation_split=0.2
)

# Prédire
predictions = predictor.predict(new_features)
```

### 3. Calcul des Risques

```python
from src.application.services.risk_service import RiskService

risk_service = RiskService()

# VaR et Expected Shortfall
var_95 = risk_service.calculate_var(
    portfolio_returns,
    confidence_level=0.95
)

es_95 = risk_service.calculate_expected_shortfall(
    portfolio_returns,
    confidence_level=0.95
)

# Stress testing
stress_results = risk_service.stress_test_portfolio(
    portfolio,
    scenarios=['market_crash', 'interest_rate_shock']
)
```

### 4. Backtesting

```python
from src.application.use_cases.backtest_strategy import BacktestStrategyUseCase

backtest_use_case = BacktestStrategyUseCase(...)

results = backtest_use_case.execute(
    strategy='mean_reversion',
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_capital=100000
)

print(f"Return total: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
```

### 5. Monitoring & Drift Detection

```python
from src.infrastructure.ml.monitoring.drift_detector import DriftDetector

drift_detector = DriftDetector()

# Détecter le drift
drift_report = drift_detector.detect_drift(
    reference_data=historical_features,
    current_data=recent_features
)

if drift_report.drift_detected:
    print("⚠️ Drift détecté! Retraining recommandé.")
    # Déclencher retraining automatique
```

## 🧪 Tests

```bash
# Tous les tests
pytest

# Tests unitaires uniquement
pytest tests/unit/

# Tests d'intégration
pytest tests/integration/

# Coverage
pytest --cov=src --cov-report=html
```

## 📈 MLOps Pipeline

### 1. Training Pipeline (Airflow)

```python
# mlops/airflow/dags/model_training_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator

dag = DAG(
    'model_training_pipeline',
    schedule_interval='@daily',
    ...
)

tasks:
1. Fetch market data
2. Engineer features  
3. Train models
4. Validate performance
5. Register in MLflow
6. Deploy if performance > threshold
```

### 2. Monitoring Dashboard (Grafana)

- Drift metrics
- Model performance
- Prediction errors
- Portfolio metrics
- System health

### 3. Retraining Trigger

Retraining automatique si:
- Drift détecté (> threshold)
- Performance dégradée (> threshold)
- Nouvelle donnée disponible (schedule)

## 📚 Documentation

Documentation complète dans `/docs`:

- [Architecture Decision Records (ADR)](docs/adr/)
- [API Documentation](docs/api/)
- [Model Cards](docs/model_cards/)
- [Compliance & Regulatory](docs/compliance/)

## 🔧 Configuration

Variables d'environnement clés dans `.env`:

```bash
# Databases
MONGODB_URI=mongodb://localhost:27017
REDIS_HOST=localhost

# Optimization
OPTIMIZATION_SOLVER=GUROBI  # ou ORTOOLS
OPTIMIZATION_TIMEOUT=300

# Risk Management
MAX_POSITION_SIZE=0.15
MAX_SECTOR_EXPOSURE=0.30

# Model Monitoring
DRIFT_DETECTION_THRESHOLD=0.1
RETRAINING_THRESHOLD=0.15
```

## 🎓 Concepts Financiers Implémentés

### Théorie Moderne du Portefeuille
- **Markowitz Optimization**: Frontière efficiente
- **CAPM**: Beta, Alpha, Sharpe Ratio
- **Black-Litterman**: Incorporation de vues subjectives

### Gestion des Risques
- **VaR**: Value at Risk (historique, paramétrique, Monte Carlo)
- **ES/CVaR**: Expected Shortfall
- **Stress Testing**: Scénarios de crise
- **Risk Parity**: Égalisation des contributions au risque

### Trading & Exécution
- **Rebalancing**: Seuils optimaux
- **Transaction Costs**: Impact sur la performance
- **Slippage Modeling**: Simulation réaliste

## 🤝 Contribution

Pour contribuer:

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amazing-feature`)
3. Commit (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

## 📝 License

MIT License

## 👨‍💻 Auteur

Développé pour démonstration de compétences en:
- Data Science & ML
- Optimisation mathématique
- MLOps & Software Engineering
- Finance quantitative

---

**Note**: Projet académique/portfolio pour candidature Data Scientist en Banque.
