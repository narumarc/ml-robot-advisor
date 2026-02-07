# Guide de Démarrage Rapide

## Installation en 5 minutes

### 1. Cloner et installer

```bash
# Cloner le projet
git clone <your-repo-url>
cd robo-advisor-project

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer le projet
make install
# OU
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copier le fichier d'environnement
cp .env.example .env

# Éditer avec vos clés API (optionnel pour la démo)
nano .env
```

### 3. Lancer l'infrastructure

```bash
# Démarrer MongoDB, Redis, MLflow, etc.
make docker-up

# Attendre que tous les services soient prêts (~30 secondes)
```

### 4. Lancer la démonstration

```bash
# Exécuter le script de démonstration complet
python scripts/demo_complete.py
```

Vous verrez:
- ✅ Extraction des données de marché
- ✅ Feature engineering
- ✅ Prédictions ML
- ✅ Optimisation (Markowitz, Risk Parity, CVaR)
- ✅ Calcul des risques (VaR, ES)
- ✅ Backtesting
- ✅ Monitoring & drift detection

## Utilisation de l'API

### Démarrer l'API

```bash
make run-api
# OU
uvicorn src.presentation.api.main:app --reload
```

L'API sera accessible sur `http://localhost:8000`

### Documentation interactive

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Exemples de requêtes

#### 1. Optimiser un portefeuille

```bash
curl -X POST "http://localhost:8000/api/v1/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "assets": [
      {"ticker": "AAPL", "expected_return": 0.12},
      {"ticker": "MSFT", "expected_return": 0.10},
      {"ticker": "GOOGL", "expected_return": 0.15}
    ],
    "method": "markowitz",
    "risk_free_rate": 0.02,
    "max_position_size": 0.15
  }'
```

Réponse:
```json
{
  "weights": {
    "AAPL": 0.35,
    "MSFT": 0.30,
    "GOOGL": 0.35
  },
  "expected_return": 0.123,
  "expected_risk": 0.18,
  "sharpe_ratio": 0.572,
  "status": "optimal",
  "method": "markowitz"
}
```

#### 2. Calculer les métriques de risque

```bash
curl -X POST "http://localhost:8000/api/v1/risk" \
  -H "Content-Type: application/json" \
  -d '{
    "confidence_level": 0.95
  }'
```

Réponse:
```json
{
  "var": -0.025,
  "es": -0.032,
  "volatility": 0.18,
  "max_drawdown": -0.15
}
```

#### 3. Prédire les rendements

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT", "GOOGL"],
    "model_type": "xgboost",
    "horizon": 1
  }'
```

## Commandes Make utiles

```bash
# Tests
make test              # Tous les tests
make test-unit         # Tests unitaires
make test-integration  # Tests d'intégration

# Code Quality
make lint              # Linter (flake8, mypy)
make format            # Formatter (black, isort)

# ML Operations
make train-models      # Entraîner les modèles
make optimize          # Optimiser un portefeuille
make backtest          # Lancer un backtest

# Monitoring
make monitor           # Ouvrir Grafana
make mlflow-ui         # Ouvrir MLflow UI

# Docker
make docker-up         # Démarrer les services
make docker-down       # Arrêter les services
make docker-logs       # Voir les logs
```

## Workflows courants

### Workflow 1: Optimisation de portefeuille

```python
# scripts/optimize_example.py
import pandas as pd
from src.infrastructure.optimization.portfolio_optimizer import PortfolioOptimizer

# Données
expected_returns = pd.Series({
    'AAPL': 0.12,
    'MSFT': 0.10,
    'GOOGL': 0.15
})

covariance_matrix = pd.DataFrame(...)  # Votre matrice de covariance

# Optimisation
optimizer = PortfolioOptimizer()
result = optimizer.optimize_markowitz(
    expected_returns=expected_returns,
    covariance_matrix=covariance_matrix,
    max_position_size=0.15
)

print(f"Poids optimaux: {result.weights}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.4f}")
```

### Workflow 2: Entraîner un modèle ML

```python
# scripts/train_example.py
from src.infrastructure.ml.models.return_predictor import ReturnPredictor
import yfinance as yf

# Télécharger les données
data = yf.download('AAPL', start='2020-01-01', end='2023-12-31')

# Créer et entraîner le modèle
predictor = ReturnPredictor(model_type='xgboost')
features = predictor.prepare_features(data['Close'])

# Target: rendement futur sur 1 jour
y = data['Close'].pct_change().shift(-1)

# Entraîner
metrics = predictor.train(features, y, validation_split=0.2)
print(f"Validation MSE: {metrics['val_mse']:.6f}")

# Sauvegarder
predictor.save('models/return_predictor.pkl')
```

### Workflow 3: Backtesting

```python
# scripts/backtest_example.py
import pandas as pd
import numpy as np

# Simuler une stratégie
returns = ...  # Vos rendements de portefeuille

# Métriques
total_return = (1 + returns).prod() - 1
sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)

# Drawdown
cumulative = (1 + returns).cumprod()
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min()

print(f"Rendement total: {total_return:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
```

## Dashboards & Monitoring

### MLflow (Expérimentations ML)

```bash
# Ouvrir MLflow UI
make mlflow-ui
# OU
open http://localhost:5000
```

Vous y trouverez:
- Historique des entraînements
- Métriques de performance
- Paramètres des modèles
- Artefacts (modèles sauvegardés)

### Grafana (Monitoring système)

```bash
# Ouvrir Grafana
make monitor
# OU
open http://localhost:3000
# Login: admin / admin
```

Dashboards disponibles:
- Performance des modèles
- Drift detection
- Métriques de portefeuille
- Santé du système

## Troubleshooting

### Problème: Services Docker ne démarrent pas

```bash
# Vérifier les logs
docker-compose logs

# Redémarrer les services
make docker-down
make docker-up
```

### Problème: Erreur d'import Python

```bash
# Vérifier que vous êtes dans le bon environnement
which python

# Réinstaller les dépendances
pip install -r requirements.txt
```

### Problème: Gurobi license non trouvée

Si vous n'avez pas de licence Gurobi:

1. Utilisez OR-Tools à la place:
```python
# Dans .env
OPTIMIZATION_SOLVER=ORTOOLS
```

2. Ou utilisez la version démo de Gurobi (limitée)

## Prochaines étapes

1. **Personnaliser les stratégies**
   - Modifier `src/infrastructure/optimization/portfolio_optimizer.py`
   - Ajouter vos propres contraintes

2. **Entraîner vos modèles**
   - Utiliser vos données historiques
   - Tester différents modèles (LSTM, Transformer)

3. **Déployer en production**
   - Configurer un serveur
   - Utiliser Kubernetes pour scalabilité
   - Ajouter monitoring avancé

4. **Explorer les notebooks**
   - `notebooks/optimization_examples.ipynb`
   - `notebooks/ml_experiments.ipynb`
   - `notebooks/risk_analysis.ipynb`

## Ressources

- [Documentation complète](docs/)
- [Architecture](docs/ARCHITECTURE.md)
- [API Reference](http://localhost:8000/docs)
- [Contributing Guide](CONTRIBUTING.md)

## Support

Pour toute question:
- Ouvrir une issue sur GitHub
- Consulter la documentation
- Contacter l'équipe

---

**Bon trading! 📈💼**
