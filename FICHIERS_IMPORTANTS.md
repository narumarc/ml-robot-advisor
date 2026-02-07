# 📋 Liste des Fichiers Importants

## 📚 Documentation (à lire en premier!)

1. **README_FINAL.md** - Vue d'ensemble complète du projet
2. **INSTALLATION.md** - Guide d'installation détaillé
3. **docs/ARCHITECTURE.md** - Architecture Clean Architecture + DDD
4. **docs/PORTS_AND_ADAPTERS.md** - Explication Hexagonal Architecture
5. **docs/ML_VS_MLOPS.md** - Distinction ML vs MLOps
6. **ML_INFRASTRUCTURE_COMPLETE.md** - Documentation ML
7. **MLOPS_COMPLETE.md** - Documentation MLOps
8. **docs/QUICKSTART.md** - Quick start guide
9. **TECHNOLOGIES.md** - Technologies utilisées

---

## 🏗️ Architecture - Domain Layer

### Entities (Core Business)
- `src/domain/entities/portfolio.py` - Portfolio aggregate root
- `src/domain/entities/asset.py` - Asset entity
- `src/domain/entities/position.py` - Position entity
- `src/domain/value_objects/risk_metrics.py` - Risk metrics VO

### Ports (Interfaces)
- `src/domain/ports/repositories/portfolio_repository_interface.py`
  - IPortfolioRepository
  - IAssetRepository
  - ITransactionRepository
  
- `src/domain/ports/services/external_services_interface.py`
  - IMarketDataService
  - IFeatureStoreService
  - ICacheService

- `src/domain/ports/ml/ml_services_interface.py`
  - IReturnPredictor
  - IVolatilityPredictor
  - IDriftDetector

---

## 🔧 Infrastructure - Adapters

### Persistence (MongoDB, Redis)
- `src/infrastructure/persistence/mongodb/portfolio_repository.py`
  - MongoDBPortfolioRepository
  - MongoDBAssetRepository
  - MongoDBTransactionRepository

- `src/infrastructure/persistence/redis/feature_store.py`
  - RedisFeatureStore
  - RedisCacheService
  - RedisPriceCache

### Data Sources
- `src/infrastructure/data_sources/market_data.py`
  - YFinanceDataSource
  - AlphaVantageDataSource

### ETL Pipeline
- `src/infrastructure/etl/extractors/market_data_extractor.py`
  - MarketDataExtractor
  - BatchExtractor

### Optimization
- `src/infrastructure/optimization/portfolio_optimizer.py`
  - GurobiOptimizer
  - MarkowitzOptimization
  - RiskParityOptimization
  - CVaROptimization

### Risk Management
- `src/infrastructure/risk_management/risk_calculator.py`
  - RiskCalculator (VaR, ES, Sharpe, Sortino, etc.)
  - StressTester

---

## 🤖 Machine Learning Infrastructure

### Models
- `src/infrastructure/ml/models/return_predictor.py`
  - XGBoostReturnPredictor
  - LightGBMReturnPredictor
  - LSTMReturnPredictor

- `src/infrastructure/ml/models/volatility_predictor.py`
  - GARCHVolatilityPredictor
  - MLVolatilityPredictor
  - EWMAVolatilityPredictor

- `src/infrastructure/ml/models/anomaly_detector.py`
  - IsolationForestDetector

### Training
- `src/infrastructure/ml/training/trainer.py` - ModelTrainer avec MLflow
- `src/infrastructure/ml/training/cross_validator.py` - Time series CV

### Monitoring
- `src/infrastructure/ml/monitoring/drift_detector.py` - Evidently drift
- `src/infrastructure/ml/monitoring/data_quality_checker.py` - Data quality
- `src/infrastructure/ml/monitoring/model_performance_tracker.py`

### Preprocessing
- `src/infrastructure/ml/preprocessing/feature_engineer.py`
  - FinancialFeatureEngineer (technical indicators)

### Evaluation
- `src/infrastructure/ml/evaluation/regression_metrics.py`
- `src/infrastructure/ml/evaluation/financial_metrics.py`

---

## 🚀 MLOps Scripts (Exécutables)

### Training Scripts
- `mlops/training/train_return_predictor.py` - Train return model
- `mlops/training/train_volatility_model.py` - Train volatility model
- `mlops/training/train_all_models.py` - Master training script

### Monitoring Scripts
- `mlops/monitoring/check_drift.py` - Drift detection
- `mlops/monitoring/check_data_quality.py` - Data quality checks
- `mlops/monitoring/check_performance.py` - Performance monitoring
- `mlops/monitoring/generate_reports.py` - HTML reports

### Retraining Scripts
- `mlops/retraining/auto_retrain_pipeline.py` - Auto retraining
- `mlops/retraining/retrain_trigger.py` - Trigger logic

### Deployment Scripts
- `mlops/deployment/deploy_model.py` - Deploy to production
- `mlops/deployment/rollback_model.py` - Rollback version
- `mlops/deployment/model_versioning.py` - Model registry

### Airflow DAGs
- `mlops/airflow/dags/daily_training_dag.py` - Daily training (2 AM)
- `mlops/airflow/dags/drift_monitoring_dag.py` - Drift monitoring (6h)
- `mlops/airflow/dags/retraining_dag.py` - Auto retraining

---

## 🌐 API & Presentation

### FastAPI
- `src/presentation/api/main.py` - FastAPI application
- `src/presentation/api/routers/` - API routers

---

## ⚙️ Configuration

- `config/settings.py` - Pydantic settings
- `config/logging_config.py` - Logging configuration
- `.env.example` - Environment variables template
- `requirements.txt` - Python dependencies
- `docker-compose.yml` - Docker services
- `Makefile` - Convenience commands

---

## 🧪 Tests

- `tests/unit/test_domain_entities.py` - Domain tests
- `tests/integration/` - Integration tests
- `pytest.ini` - Pytest configuration

---

## 📜 Scripts Utiles

- `scripts/demo_complete.py` - Démo complète du système
- `scripts/init_db.py` - Initialize database

---

## 📊 Combien de Fichiers?

### Par Catégorie:

**Documentation:** 9 fichiers MD
- README_FINAL.md
- INSTALLATION.md
- docs/ (7 fichiers)

**Domain Layer:** 7 fichiers
- entities/ (3)
- ports/ (3 interfaces)
- value_objects/ (1)

**Infrastructure:** 20+ fichiers
- persistence/ (3)
- ml/ (12)
- optimization/ (1)
- risk_management/ (1)
- data_sources/ (1)
- etl/ (2)

**MLOps Scripts:** 15 fichiers
- training/ (3)
- monitoring/ (4)
- retraining/ (2)
- deployment/ (3)
- airflow/dags/ (3)

**Configuration:** 6 fichiers
- config/ (2)
- .env.example
- requirements.txt
- docker-compose.yml
- Makefile

**Tests:** 3+ fichiers
- pytest.ini
- tests/unit/
- tests/integration/

**Total: ~60-70 fichiers Python + 15 fichiers de config/doc**

---

## 🎯 Fichiers à Voir en Premier

Pour comprendre le projet rapidement:

1. **README_FINAL.md** - Start here!
2. **docs/ARCHITECTURE.md** - Comprendre l'architecture
3. **src/domain/entities/portfolio.py** - Core business logic
4. **src/infrastructure/optimization/portfolio_optimizer.py** - Gurobi optimization
5. **mlops/training/train_return_predictor.py** - ML training example
6. **scripts/demo_complete.py** - Working demo

---

## 📦 Structure Finale

```
robo-advisor-project/          (Root)
├── README_FINAL.md            ⭐ START HERE
├── INSTALLATION.md            ⭐ Installation guide
├── FICHIERS_IMPORTANTS.md     ⭐ Ce fichier
│
├── docs/                      📚 Documentation
│   ├── ARCHITECTURE.md
│   ├── PORTS_AND_ADAPTERS.md
│   ├── ML_VS_MLOPS.md
│   └── QUICKSTART.md
│
├── src/                       🏗️ Source Code
│   ├── domain/               (Core Business)
│   ├── application/          (Use Cases)
│   ├── infrastructure/       (Adapters)
│   └── presentation/         (API)
│
├── mlops/                     🚀 MLOps Scripts
│   ├── training/
│   ├── monitoring/
│   ├── retraining/
│   ├── deployment/
│   └── airflow/dags/
│
├── config/                    ⚙️ Configuration
├── tests/                     🧪 Tests
├── scripts/                   📜 Utility scripts
└── docker/                    🐳 Docker files
```

---

## ✅ Checklist pour Utiliser le Projet

- [ ] Lire README_FINAL.md
- [ ] Suivre INSTALLATION.md
- [ ] Comprendre l'architecture (docs/ARCHITECTURE.md)
- [ ] Regarder le code domain layer
- [ ] Tester un script MLOps
- [ ] Lancer la démo (scripts/demo_complete.py)
- [ ] Explorer l'API FastAPI
- [ ] Lire la doc MLOps
- [ ] Personnaliser pour ton usage

---

**Bon courage ! 🚀**
