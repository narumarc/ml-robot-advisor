# Architecture Documentation

## Vue d'ensemble

Ce projet implémente une plateforme complète de **Robo-Advisor** pour l'optimisation de portefeuille avec ML, optimisation mathématique, et MLOps, en suivant les principes de **Clean Architecture** et **Domain-Driven Design (DDD)**.

## Principes architecturaux

### 1. Clean Architecture

L'architecture est organisée en couches concentriques avec dépendances unidirectionnelles vers l'intérieur:

```
┌─────────────────────────────────────────────────────┐
│         Infrastructure Layer (External)             │
│  - Database (MongoDB, Redis)                        │
│  - External APIs (Market Data, ML Models)           │
│  - Frameworks (FastAPI, Airflow)                    │
└────────────────┬────────────────────────────────────┘
                 │ depends on ↓
┌────────────────┴────────────────────────────────────┐
│              Presentation Layer                     │
│  - REST API (FastAPI endpoints)                     │
│  - CLI interfaces                                   │
│  - Response DTOs                                    │
└────────────────┬────────────────────────────────────┘
                 │ depends on ↓
┌────────────────┴────────────────────────────────────┐
│            Application Layer                        │
│  - Use Cases (OptimizePortfolio, PredictReturns)   │
│  - Application Services                             │
│  - DTOs (Request/Response)                          │
└────────────────┬────────────────────────────────────┘
                 │ depends on ↓
┌────────────────┴────────────────────────────────────┐
│               Domain Layer (Core)                   │
│  - Entities (Portfolio, Asset)                      │
│  - Value Objects (Money, ReturnRate)                │
│  - Aggregates (PortfolioAggregate)                  │
│  - Domain Services                                  │
│  - Repository Interfaces                            │
└─────────────────────────────────────────────────────┘
```

### 2. Domain-Driven Design (DDD)

#### Bounded Contexts

Le système est organisé en contextes délimités:

1. **Portfolio Management Context**
   - Gestion des portefeuilles
   - Positions et transactions
   - Rebalancing

2. **Market Data Context**
   - Prix des actifs
   - Données historiques
   - Indicateurs techniques

3. **Risk Management Context**
   - Calcul des métriques de risque
   - Stress testing
   - Monitoring des limites

4. **ML/Prediction Context**
   - Entraînement des modèles
   - Prédictions
   - Monitoring du drift

#### Entités et Aggregates

**Portfolio Aggregate (Root)**
```
Portfolio (Aggregate Root)
├── Positions (Entities)
├── Transactions (Entities)
└── RiskMetrics (Value Objects)
```

**Asset (Entity)**
- Identifié par ID unique
- Propriétés: ticker, type, sector
- Comportements: update_price, update_risk_metrics

#### Value Objects

- **Money**: Représente une valeur monétaire
- **ReturnRate**: Taux de rendement
- **RiskLevel**: Niveau de risque

#### Domain Services

Services qui ne s'appartiennent pas naturellement à une entité:

- **PortfolioDomainService**: Logique métier complexe
- **RiskCalculationService**: Calculs de risque partagés

### 3. Dependency Inversion

Les couches externes dépendent des abstractions définies dans le domaine:

```python
# Domain layer defines interface
class PortfolioRepository(ABC):
    @abstractmethod
    async def save(self, portfolio: Portfolio) -> None:
        pass
    
    @abstractmethod
    async def find_by_id(self, portfolio_id: UUID) -> Optional[Portfolio]:
        pass

# Infrastructure layer implements interface
class MongoDBPortfolioRepository(PortfolioRepository):
    async def save(self, portfolio: Portfolio) -> None:
        # MongoDB-specific implementation
        ...
```

## Structure des répertoires

```
robo-advisor-project/
├── src/
│   ├── domain/                    # Couche Domain (DDD)
│   │   ├── entities/              # Entités du domaine
│   │   │   ├── portfolio.py
│   │   │   ├── asset.py
│   │   │   └── transaction.py
│   │   ├── value_objects/         # Value Objects
│   │   │   ├── money.py
│   │   │   └── return_rate.py
│   │   ├── aggregates/            # Aggregates
│   │   │   └── portfolio_aggregate.py
│   │   ├── repositories/          # Interfaces Repository
│   │   │   └── portfolio_repository.py
│   │   └── services/              # Domain Services
│   │       └── portfolio_domain_service.py
│   │
│   ├── application/               # Couche Application
│   │   ├── use_cases/             # Use Cases (CQRS)
│   │   │   ├── optimize_portfolio.py
│   │   │   ├── predict_returns.py
│   │   │   └── rebalance_portfolio.py
│   │   ├── dto/                   # Data Transfer Objects
│   │   │   └── portfolio_dto.py
│   │   └── services/              # Application Services
│   │       ├── ml_service.py
│   │       └── optimization_service.py
│   │
│   ├── infrastructure/            # Couche Infrastructure
│   │   ├── database/              # Persistance
│   │   │   ├── mongodb_client.py
│   │   │   └── repositories/
│   │   ├── ml/                    # Machine Learning
│   │   │   ├── models/
│   │   │   ├── training/
│   │   │   └── monitoring/
│   │   ├── optimization/          # Optimisation (Gurobi)
│   │   │   └── portfolio_optimizer.py
│   │   └── external_apis/         # APIs externes
│   │
│   └── presentation/              # Couche Présentation
│       ├── api/                   # REST API
│       │   └── main.py
│       └── cli/                   # CLI
│
├── tests/                         # Tests
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── config/                        # Configuration
├── scripts/                       # Scripts utilitaires
├── docs/                          # Documentation
└── mlops/                         # MLOps pipelines
```

## Flux de données

### 1. Optimisation de Portefeuille

```
[API Request]
     ↓
[OptimizePortfolioUseCase] (Application)
     ↓
[Fetch Portfolio] → [PortfolioRepository] → [MongoDB]
     ↓
[Fetch Market Data] → [MarketDataService] → [External API]
     ↓
[Predict Returns] → [MLService] → [Return Predictor Model]
     ↓
[Optimize] → [OptimizationService] → [Gurobi]
     ↓
[Update Portfolio] → [PortfolioRepository] → [MongoDB]
     ↓
[API Response]
```

### 2. Prédiction ML

```
[Scheduled Job] (Airflow)
     ↓
[Fetch Market Data] → [ETL Pipeline]
     ↓
[Feature Engineering] → [Feature Store] (Redis)
     ↓
[Train Model] → [MLflow]
     ↓
[Evaluate] → [Evidently] (Drift Detection)
     ↓
[Register Model] → [MLflow Model Registry]
     ↓
[Deploy if Valid]
```

## Patterns de conception

### 1. Repository Pattern

Abstraction de la persistance des données:

```python
class PortfolioRepository(ABC):
    """Interface for portfolio persistence."""
    
    @abstractmethod
    async def save(self, portfolio: Portfolio) -> None:
        """Save portfolio."""
        
    @abstractmethod
    async def find_by_id(self, id: UUID) -> Optional[Portfolio]:
        """Find portfolio by ID."""
```

### 2. Use Case Pattern

Encapsulation de la logique métier:

```python
class OptimizePortfolioUseCase:
    """Use case for portfolio optimization."""
    
    def __init__(
        self,
        portfolio_repo: PortfolioRepository,
        ml_service: MLService,
        optimizer: PortfolioOptimizer
    ):
        self.portfolio_repo = portfolio_repo
        self.ml_service = ml_service
        self.optimizer = optimizer
    
    async def execute(self, request: OptimizeRequest) -> OptimizeResponse:
        # Use case logic
        ...
```

### 3. Factory Pattern

Création d'objets complexes:

```python
class PortfolioFactory:
    """Factory for creating portfolios."""
    
    @staticmethod
    def create_balanced_portfolio(owner_id: str) -> Portfolio:
        return Portfolio(
            name="Balanced Portfolio",
            owner_id=owner_id,
            strategy="balanced",
            max_position_size=0.15
        )
```

### 4. Strategy Pattern

Algorithmes d'optimisation interchangeables:

```python
class OptimizationStrategy(ABC):
    @abstractmethod
    def optimize(self, data: OptimizationData) -> OptimizationResult:
        pass

class MarkowitzStrategy(OptimizationStrategy):
    def optimize(self, data: OptimizationData) -> OptimizationResult:
        # Markowitz implementation
        ...

class RiskParityStrategy(OptimizationStrategy):
    def optimize(self, data: OptimizationData) -> OptimizationResult:
        # Risk Parity implementation
        ...
```

## Principes SOLID

### Single Responsibility Principle (SRP)
Chaque classe a une seule responsabilité:
- `Asset`: Gérer les informations d'un actif
- `Portfolio`: Gérer un portefeuille
- `PortfolioOptimizer`: Optimiser les poids

### Open/Closed Principle (OCP)
Ouvert à l'extension, fermé à la modification:
- Nouvelles stratégies d'optimisation via Strategy Pattern
- Nouveaux modèles ML via interfaces

### Liskov Substitution Principle (LSP)
Les implémentations peuvent remplacer les abstractions:
- `MongoDBPortfolioRepository` remplace `PortfolioRepository`

### Interface Segregation Principle (ISP)
Interfaces spécifiques et minimales:
- `PortfolioRepository` ne contient que les méthodes nécessaires

### Dependency Inversion Principle (DIP)
Dépendance sur des abstractions:
- Use cases dépendent de `PortfolioRepository` (interface)
- Pas de dépendance directe sur MongoDB

## Testing Strategy

### Pyramide des tests

```
      /\
     /E2E\        ← End-to-End (10%)
    /______\
   /Integ. \      ← Integration (30%)
  /__________\
 /   Unit     \   ← Unit Tests (60%)
/______________\
```

### Types de tests

1. **Unit Tests** (60%)
   - Entities, Value Objects
   - Domain Services
   - Pure functions

2. **Integration Tests** (30%)
   - Repositories avec database
   - External APIs
   - ML pipelines

3. **E2E Tests** (10%)
   - API endpoints
   - Complete workflows

## Performance & Scalability

### Caching Strategy

```
Redis (Feature Store)
├── Asset prices (TTL: 1min)
├── Technical indicators (TTL: 5min)
└── Model predictions (TTL: 1hour)
```

### Optimization

- **Database Indexing**: MongoDB indexes sur ticker, date
- **Query Optimization**: Projection, limit, pagination
- **Async/Await**: Operations I/O asynchrones
- **Batch Processing**: Traitement par lots pour ML

### Monitoring

- **Prometheus**: Métriques système et application
- **Grafana**: Dashboards de visualisation
- **MLflow**: Tracking des expériences ML
- **Evidently**: Monitoring du drift des modèles

## Security

### Authentication & Authorization
- JWT tokens pour API
- Role-based access control (RBAC)
- API rate limiting

### Data Security
- Encryption at rest (MongoDB)
- Encryption in transit (TLS)
- Secrets management (environment variables)

### Compliance
- GDPR compliance (data retention policies)
- Financial regulations (audit logs)
- Model documentation (for regulatory review)

## Conclusion

Cette architecture garantit:
- ✅ Maintenabilité: Code modulaire et testable
- ✅ Scalabilité: Architecture découplée
- ✅ Testabilité: Isolation des composants
- ✅ Flexibilité: Facile d'ajouter de nouvelles fonctionnalités
- ✅ Performance: Optimisations et caching
