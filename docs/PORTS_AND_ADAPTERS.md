# Architecture Hexagonale - PORTS & ADAPTERS

## Vue d'ensemble

Cette architecture implémente le pattern **Hexagonal Architecture** (Ports & Adapters) d'Alistair Cockburn, aussi appelé **Clean Architecture**.

```
┌─────────────────────────────────────────────────────┐
│                  ADAPTERS (Out)                     │
│  - MongoDB, Redis                                   │
│  - yFinance, Alpha Vantage                          │
│  - Email, SMS                                       │
└────────────────┬────────────────────────────────────┘
                 │ implements ↓
┌────────────────┴────────────────────────────────────┐
│                     PORTS                           │
│  Interfaces defining contracts:                    │
│  - IPortfolioRepository                             │
│  - IMarketDataService                               │
│  - IFeatureStoreService                             │
│  - IReturnPredictor                                 │
└────────────────┬────────────────────────────────────┘
                 │ used by ↓
┌────────────────┴────────────────────────────────────┐
│              DOMAIN (Core Business)                 │
│  - Entities (Portfolio, Asset)                     │
│  - Value Objects (Money, ReturnRate)                │
│  - Aggregates                                       │
│  - Domain Services                                  │
└─────────────────────────────────────────────────────┘
```

## PORTS (Interfaces)

Les **PORTS** sont des interfaces qui définissent les contrats entre le domaine et l'infrastructure.

### Repository Ports

#### `IPortfolioRepository`
```python
class IPortfolioRepository(ABC):
    @abstractmethod
    async def save(self, portfolio: Portfolio) -> None: pass
    
    @abstractmethod
    async def find_by_id(self, portfolio_id: UUID) -> Optional[Portfolio]: pass
    
    @abstractmethod
    async def find_by_owner(self, owner_id: str) -> List[Portfolio]: pass
```

**Pourquoi?** Permet de changer de base de données (MongoDB → PostgreSQL) sans toucher au domaine.

#### `IAssetRepository`
```python
class IAssetRepository(ABC):
    @abstractmethod
    async def find_by_ticker(self, ticker: str) -> Optional[Asset]: pass
    
    @abstractmethod
    async def update_price(self, ticker: str, price: float, timestamp: datetime) -> None: pass
```

### Service Ports

#### `IMarketDataService`
```python
class IMarketDataService(ABC):
    @abstractmethod
    async def get_current_price(self, ticker: str) -> float: pass
    
    @abstractmethod
    async def get_historical_prices(
        self, ticker: str, start: datetime, end: datetime
    ) -> pd.DataFrame: pass
```

**Pourquoi?** Permet de changer de fournisseur (yFinance → Bloomberg) sans impact sur le domaine.

#### `IFeatureStoreService`
```python
class IFeatureStoreService(ABC):
    @abstractmethod
    async def save_features(self, key: str, features: pd.DataFrame, ttl: int) -> None: pass
    
    @abstractmethod
    async def get_features(self, key: str) -> Optional[pd.DataFrame]: pass
```

### ML Ports

#### `IReturnPredictor`
```python
class IReturnPredictor(ABC):
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]: pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> PredictionResult: pass
```

**Pourquoi?** Permet de changer de modèle (XGBoost → LSTM) sans changer l'application.

#### `IDriftDetector`
```python
class IDriftDetector(ABC):
    @abstractmethod
    def detect_drift(
        self, reference_data: pd.DataFrame, current_data: pd.DataFrame
    ) -> DriftReport: pass
```

## ADAPTERS (Implémentations)

Les **ADAPTERS** sont les implémentations concrètes des ports.

### Infrastructure Adapters

#### MongoDB Adapter
```
src/infrastructure/persistence/mongodb/
├── portfolio_repository.py    # MongoDBPortfolioRepository
├── asset_repository.py         # MongoDBAssetRepository
└── transaction_repository.py   # MongoDBTransactionRepository
```

**Implémente:** `IPortfolioRepository`, `IAssetRepository`

**Exemple:**
```python
class MongoDBPortfolioRepository(IPortfolioRepository):
    def __init__(self, database: AsyncIOMotorDatabase):
        self.db = database
        self.collection = self.db.portfolios
    
    async def save(self, portfolio: Portfolio) -> None:
        portfolio_dict = portfolio.to_dict()
        await self.collection.update_one(
            {"id": portfolio_dict["id"]},
            {"$set": portfolio_dict},
            upsert=True
        )
```

#### Redis Adapter
```
src/infrastructure/persistence/redis/
├── feature_store.py      # RedisFeatureStore
├── cache_service.py      # RedisCacheService
└── price_cache.py        # RedisPriceCache
```

**Implémente:** `IFeatureStoreService`, `ICacheService`

**Exemple:**
```python
class RedisFeatureStore(IFeatureStoreService):
    async def save_features(self, key: str, features: pd.DataFrame, ttl: int) -> None:
        serialized = pickle.dumps(features)
        await self.redis.setex(key, ttl, serialized)
```

### Data Source Adapters

#### yFinance Adapter
```
src/infrastructure/data_sources/
└── market_data.py
    ├── YFinanceDataSource       # IMarketDataService
    └── AlphaVantageDataSource   # IMarketDataService
```

**Implémente:** `IMarketDataService`

**Exemple:**
```python
class YFinanceDataSource(IMarketDataService):
    async def get_current_price(self, ticker: str) -> float:
        stock = yf.Ticker(ticker)
        return float(stock.info['currentPrice'])
```

### ML Adapters

#### XGBoost Predictor
```
src/infrastructure/ml/models/
├── return_predictor.py
│   ├── XGBoostPredictor      # IReturnPredictor
│   ├── LSTMPredictor         # IReturnPredictor
│   └── LightGBMPredictor     # IReturnPredictor
```

**Implémente:** `IReturnPredictor`

## Dependency Injection

L'injection de dépendances connecte les ports et adapters:

```python
# Infrastructure layer creates adapters
mongo_uri = "mongodb://localhost:27017"
database = AsyncIOMotorClient(mongo_uri).robo_advisor
portfolio_repo = MongoDBPortfolioRepository(database)

# Application layer receives through ports
optimize_use_case = OptimizePortfolioUseCase(
    portfolio_repository=portfolio_repo,  # Port, not concrete class!
    market_data_service=YFinanceDataSource(),
    optimizer=GurobiOptimizer()
)
```

## Avantages de cette Architecture

### 1. **Testabilité**
```python
# Easy to mock dependencies
class MockPortfolioRepository(IPortfolioRepository):
    async def save(self, portfolio): pass
    async def find_by_id(self, id): return mock_portfolio

# Test avec mock
use_case = OptimizePortfolioUseCase(
    portfolio_repository=MockPortfolioRepository()
)
```

### 2. **Flexibilité**
Changer d'implémentation sans toucher au domaine:

```python
# Before: MongoDB
portfolio_repo = MongoDBPortfolioRepository(db)

# After: PostgreSQL
portfolio_repo = PostgreSQLPortfolioRepository(db)

# Domain & Application layers unchanged!
```

### 3. **Indépendance**
Le domaine ne dépend d'aucune technologie externe:
- ❌ Pas d'import de pymongo dans le domaine
- ❌ Pas d'import de yfinance dans le domaine
- ✅ Seulement des abstractions (interfaces)

### 4. **Évolutivité**
Ajouter de nouvelles implémentations facilement:

```python
# New adapter for Bloomberg
class BloombergDataSource(IMarketDataService):
    async def get_current_price(self, ticker: str) -> float:
        # Bloomberg-specific implementation
        ...

# Use in application without changing anything else
market_data = BloombergDataSource()
```

## Structure des Fichiers

```
src/
├── domain/                        # CORE (ne dépend de rien)
│   ├── entities/
│   ├── value_objects/
│   ├── aggregates/
│   └── ports/                     # PORTS (interfaces)
│       ├── repositories/
│       │   ├── portfolio_repository_interface.py
│       │   └── asset_repository_interface.py
│       ├── services/
│       │   └── external_services_interface.py
│       └── ml/
│           └── ml_services_interface.py
│
├── application/                   # Use Cases (utilise PORTS)
│   ├── use_cases/
│   └── services/
│
└── infrastructure/               # ADAPTERS (implémente PORTS)
    ├── persistence/
    │   ├── mongodb/             # MongoDB Adapters
    │   └── redis/               # Redis Adapters
    ├── data_sources/            # External API Adapters
    ├── ml/                      # ML Adapters
    └── etl/                     # ETL Adapters
```

## Exemple Complet: Portfolio Optimization

### 1. Port (Interface)
```python
# src/domain/ports/repositories/portfolio_repository_interface.py
class IPortfolioRepository(ABC):
    @abstractmethod
    async def find_by_id(self, id: UUID) -> Optional[Portfolio]: pass
```

### 2. Adapter (Implementation)
```python
# src/infrastructure/persistence/mongodb/portfolio_repository.py
class MongoDBPortfolioRepository(IPortfolioRepository):
    async def find_by_id(self, id: UUID) -> Optional[Portfolio]:
        doc = await self.collection.find_one({"id": str(id)})
        return Portfolio.from_dict(doc) if doc else None
```

### 3. Use Case (utilise le Port)
```python
# src/application/use_cases/optimize_portfolio.py
class OptimizePortfolioUseCase:
    def __init__(
        self,
        portfolio_repo: IPortfolioRepository,  # Dépend du PORT, pas l'adapter!
        optimizer: IPortfolioOptimizer
    ):
        self.portfolio_repo = portfolio_repo
        self.optimizer = optimizer
    
    async def execute(self, portfolio_id: UUID) -> OptimizationResult:
        # Utilise le port (interface)
        portfolio = await self.portfolio_repo.find_by_id(portfolio_id)
        
        # Business logic...
        result = self.optimizer.optimize(portfolio)
        
        return result
```

### 4. Dependency Injection (lie tout ensemble)
```python
# main.py or dependency injection container
repositories = await init_mongodb_repositories(mongo_uri, db_name)
portfolio_repo = repositories["portfolio"]  # MongoDBPortfolioRepository

use_case = OptimizePortfolioUseCase(
    portfolio_repo=portfolio_repo,
    optimizer=GurobiOptimizer()
)

# Execute
result = await use_case.execute(portfolio_id)
```

## Principes SOLID Appliqués

✅ **Single Responsibility**: Chaque adapter a une seule responsabilité  
✅ **Open/Closed**: Ouvert à l'extension (nouveaux adapters) sans modification du domaine  
✅ **Liskov Substitution**: Tous les adapters respectent le contrat du port  
✅ **Interface Segregation**: Ports spécifiques et minimaux  
✅ **Dependency Inversion**: Dépendance sur des abstractions (ports) pas des implémentations  

## Tests

### Test du Domaine (sans dépendances)
```python
def test_portfolio_add_position():
    portfolio = Portfolio(name="Test", owner_id="user1", cash=Decimal("10000"))
    # Test pure business logic, no database needed!
    assert portfolio.get_total_value() == Decimal("10000")
```

### Test d'Intégration (avec mock)
```python
@pytest.fixture
def mock_repo():
    return MockPortfolioRepository()

async def test_optimize_use_case(mock_repo):
    use_case = OptimizePortfolioUseCase(portfolio_repo=mock_repo)
    result = await use_case.execute(portfolio_id)
    assert result.success
```

### Test E2E (avec vraie base)
```python
@pytest.mark.integration
async def test_full_optimization_flow():
    # Real MongoDB
    repo = MongoDBPortfolioRepository(test_db)
    use_case = OptimizePortfolioUseCase(portfolio_repo=repo)
    result = await use_case.execute(portfolio_id)
    assert result.success
```

## Conclusion

L'architecture Hexagonale (Ports & Adapters) offre:

✅ **Séparation claire** des responsabilités  
✅ **Testabilité** maximale  
✅ **Flexibilité** pour changer d'infrastructure  
✅ **Maintenabilité** à long terme  
✅ **Indépendance** du domaine vis-à-vis des frameworks  

C'est l'architecture idéale pour des projets complexes comme notre Robo-Advisor!
