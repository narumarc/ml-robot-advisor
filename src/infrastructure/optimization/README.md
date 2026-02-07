# 🏗️ Structure des Optimizers - Guide d'Intégration

## 📁 Structure du Code

```
src/infrastructure/optimization/
├── __init__.py                 # Package initialization
├── base_solver.py              # Interface abstraite (BaseSolver)
├── highs_solver.py            # Solver HiGHS (FREE, recommandé)
├── cvxpy_solver.py            # Solver CVXPY (FREE, alternatif)
├── gurobi_solver.py           # Solver Gurobi (Commercial, optionnel)
├── solver_factory.py          # Factory Pattern
└── README.md                  # Ce fichier

Ancien code (à remplacer):
├── portfolio_optimizer.py     # ANCIEN - Gurobi uniquement
```

---

## 🎯 Changements par Rapport à l'Ancien Code

### Avant (Gurobi uniquement)

```python
# Fichier unique: portfolio_optimizer.py
from src.infrastructure.optimization.portfolio_optimizer import GurobiOptimizer

optimizer = GurobiOptimizer()
result = optimizer.optimize_markowitz(returns, cov_matrix)
```

**Problèmes:**
-  Dépend uniquement de Gurobi (licence commerciale)
-  Pas de flexibilité
- Difficile à tester sans licence
-  Code couplé au solver

### Après (Multi-solver avec Factory)

```python
# Fichiers séparés par solver
from src.infrastructure.optimization import create_optimizer

# Choisis ton solver!
optimizer = create_optimizer('highs')  # FREE
# optimizer = create_optimizer('cvxpy')  # FREE
# optimizer = create_optimizer('gurobi')  # Commercial

result = optimizer.optimize_markowitz(returns, cov_matrix)
```

**Avantages:**
- ✅ 100% gratuit avec HiGHS ou CVXPY
- ✅ Flexible (change de solver en 1 ligne)
- ✅ Testable sans licence
- ✅ Code découplé (Factory Pattern)
- ✅ Facile d'ajouter de nouveaux solvers

---

## 📦 Installation

### Option 1: HiGHS (Recommandé - FREE)

```bash
# HiGHS est inclus dans scipy >= 1.9
pip install 'scipy>=1.9.0'
```

### Option 2: CVXPY (Alternative - FREE)

```bash
pip install cvxpy
```

### Option 3: Gurobi (Commercial - Optionnel)

```bash
pip install gurobipy
# + Obtenir licence (gratuite pour académique)
# + Activer: grbgetkey YOUR_KEY
```

---

## 🚀 Quick Start

### Utilisation Basique

```python
from src.infrastructure.optimization import create_optimizer
import yfinance as yf
import pandas as pd

# 1. Charger données
tickers = ['AAPL', 'MSFT', 'GOOGL']
data = yf.download(tickers, start='2023-01-01', end='2024-01-01')['Close']
returns = data.pct_change().dropna()

# 2. Calculer statistiques
expected_returns = returns.mean() * 252  # Annualisé
cov_matrix = returns.cov() * 252

# 3. Créer optimizer (HiGHS = FREE!)
optimizer = create_optimizer('highs')

# 4. Optimiser
result = optimizer.optimize_markowitz(
    expected_returns=expected_returns,
    cov_matrix=cov_matrix,
    risk_free_rate=0.02
)

# 5. Résultats
if result.success:
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    for asset, weight in result.weights.items():
        if weight > 0.01:
            print(f"  {asset}: {weight*100:.1f}%")
```

---

## 🔄 Migration depuis l'Ancien Code

### Étape 1: Copier les Nouveaux Fichiers

```bash
# Copier la structure optimizers/ dans ton projet
cp -r optimizers/* src/infrastructure/optimization/
```

### Étape 2: Mettre à Jour les Imports

**Avant:**
```python
from src.infrastructure.optimization.portfolio_optimizer import GurobiOptimizer

optimizer = GurobiOptimizer()
```

**Après:**
```python
from src.infrastructure.optimization import create_optimizer

optimizer = create_optimizer('highs')  # ou 'cvxpy', 'gurobi'
```

### Étape 3: Vérifier les Scripts MLOps

**Scripts à mettre à jour:**
- `mlops/training/train_return_predictor.py`
- `mlops/training/train_volatility_model.py`
- Tout script qui utilise `GurobiOptimizer`

**Changement:**
```python
# AVANT
from src.infrastructure.optimization.portfolio_optimizer import GurobiOptimizer
optimizer = GurobiOptimizer()

# APRÈS
from src.infrastructure.optimization import create_optimizer
optimizer = create_optimizer('highs')  # ou lire depuis config
```

### Étape 4: Configuration

**config/optimizer_config.yaml:**
```yaml
# Solver configuration
solver: highs  # Options: highs, cvxpy, gurobi

# Optimization parameters
risk_free_rate: 0.02

# Constraints
constraints:
  max_position_size: 0.40
  min_position_size: 0.05
```

**Charger la config:**
```python
import yaml
from src.infrastructure.optimization import create_optimizer

with open('config/optimizer_config.yaml') as f:
    config = yaml.safe_load(f)

optimizer = create_optimizer(
    solver=config['solver'],
    verbose=False
)
```

---

## 📚 API Reference

### BaseSolver (Interface)

```python
class BaseSolver(ABC):
    """Interface que tous les solvers doivent implémenter"""
    
    def optimize_markowitz(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.02,
        constraints: Optional[Dict] = None
    ) -> OptimizationResult:
        """Optimisation de Markowitz"""
    
    def optimize_risk_parity(
        self,
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series
    ) -> OptimizationResult:
        """Risk Parity optimization"""
```

### OptimizationResult

```python
@dataclass
class OptimizationResult:
    success: bool                    # Optimisation réussie?
    weights: Dict[str, float]        # Poids optimaux
    expected_return: float           # Rendement espéré
    volatility: float                # Volatilité
    sharpe_ratio: float              # Ratio de Sharpe
    objective_value: float           # Valeur de la fonction objectif
    solver_time: float               # Temps de résolution (secondes)
    solver_name: str                 # Nom du solver utilisé
    message: str                     # Message de succès/erreur
    cvar: Optional[float]            # CVaR (si applicable)
```

---

## 🧪 Tests

### Tester un Solver Spécifique

```python
from src.infrastructure.optimization import create_optimizer

# Test HiGHS
solver = create_optimizer('highs', verbose=True)
result = solver.optimize_markowitz(returns, cov)

assert result.success
assert abs(sum(result.weights.values()) - 1.0) < 1e-6
assert result.sharpe_ratio > 0
```

### Tester Tous les Solvers Disponibles

```python
from src.infrastructure.optimization import SolverFactory

available_solvers = SolverFactory.list_available()
print(f"Available: {available_solvers}")

for solver_name in available_solvers:
    optimizer = create_optimizer(solver_name)
    result = optimizer.optimize_markowitz(returns, cov)
    print(f"{solver_name}: Sharpe = {result.sharpe_ratio:.2f}")
```

---

## 🎓 Comprendre les Mathématiques

Un **PDF complet** est fourni avec toutes les formulations mathématiques:

📄 **Portfolio_Optimization_Mathematics.pdf**

**Contenu:**
- Notations mathématiques
- Formulation de Markowitz (objectif, contraintes, gradient)
- Risk Parity (contribution au risque, algorithme)
- CVaR (définition, formulation)
- Métriques (Sharpe, Sortino, etc.)
- Exemples numériques complets

**Pour l'entretien:**
Lis au moins les sections 1 (Notations) et 2 (Markowitz).

---

## 💡 Pour l'Entretien

### Question: "Pourquoi pas Gurobi seulement?"

**Réponse:**
> "J'ai implémenté un Factory Pattern qui supporte plusieurs solvers 
> d'optimisation. Par défaut, j'utilise HiGHS qui est gratuit et 
> open-source via scipy. Les performances sont très compétitives 
> pour les portfolios de taille modérée. Le Factory Pattern permet 
> de basculer facilement vers Gurobi en production si besoin de 
> performance maximale, sans changer le code métier. C'est une 
> approche flexible qui évite le vendor lock-in."

### Question: "HiGHS vs Gurobi - différence?"

**Réponse:**
> "Pour des portfolios <100 actifs, HiGHS résout le problème en 
> <1 seconde, ce qui est largement suffisant. Gurobi est 5-10x 
> plus rapide mais surtout sur des problèmes très larges (1000+ 
> actifs) ou avec beaucoup de contraintes complexes. Pour la 
> majorité des cas d'usage, HiGHS est un excellent choix gratuit."

---

## 📝 Checklist de Migration

- [ ] Copier les fichiers dans `src/infrastructure/optimization/`
- [ ] Installer scipy: `pip install 'scipy>=1.9.0'`
- [ ] Mettre à jour les imports (voir section Migration)
- [ ] Créer `config/optimizer_config.yaml`
- [ ] Tester avec HiGHS sur un exemple simple
- [ ] Mettre à jour les scripts MLOps
- [ ] Tester les 2 stratégies (Markowitz, Risk Parity)
- [ ] (Optionnel) Installer CVXPY pour alternative
- [ ] Lire le PDF mathématique (au moins sections 1-2)
- [ ] Supprimer ou archiver l'ancien `portfolio_optimizer.py`

---

## 🤝 Contribution

Pour ajouter un nouveau solver:

1. Créer `nouveau_solver.py` qui hérite de `BaseSolver`
2. Implémenter `optimize_markowitz()` et `optimize_risk_parity()`
3. Ajouter le solver dans `solver_factory.py`
4. Mettre à jour `__init__.py`
5. Documenter dans ce README

---

## 📚 Ressources

**Documentation:**
- HiGHS: https://highs.dev/
- CVXPY: https://www.cvxpy.org/
- Gurobi: https://www.gurobi.com/documentation/

**Papiers de Référence:**
- Markowitz, H. (1952). "Portfolio Selection"
- Maillard et al. (2010). "On the properties of equally-weighted risk contributions portfolios"

---

**Questions ? Ouvre une issue ou contacte l'équipe ! 😊**
