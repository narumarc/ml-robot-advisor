# Architecture Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Architectural Patterns](#architectural-patterns)
3. [Domain Model](#domain-model)
4. [Technology Stack](#technology-stack)
5. [Data Flow](#data-flow)
6. [ML Pipeline](#ml-pipeline)
7. [Optimization Engine](#optimization-engine)
8. [Risk Management](#risk-management)
9. [MLOps Infrastructure](#mlops-infrastructure)
10. [Scalability & Performance](#scalability--performance)

---

## System Overview

The Robo-Advisor is an enterprise-grade portfolio management system that combines:
- **Machine Learning** for return and volatility prediction
- **Mathematical Optimization** for portfolio construction
- **Risk Management** for downside protection
- **MLOps** for continuous model improvement

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Presentation Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  REST API    │  │  Dashboard   │  │  CLI Interface       │  │
│  │  (FastAPI)   │  │  (Streamlit) │  │  (Click)             │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────────┐
│                      Application Layer                          │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Use Cases (Commands & Queries)                           │ │
│  │  - OptimizePortfolio  - CalculateVaR                      │ │
│  │  - PredictReturns     - DetectAnomalies                   │ │
│  │  - BacktestStrategy   - RebalancePortfolio                │ │
│  └───────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────────┐
│                        Domain Layer                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Entities & Value Objects                                 │  │
│  │  - Portfolio (Aggregate Root)  - Position                 │  │
│  │  - RiskMetrics                 - ReturnMetrics            │  │
│  │  - TradingSignal               - Constraints              │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │  Domain Services                                          │  │
│  │  - RiskCalculator   - PortfolioValidator                  │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │  Repository Interfaces                                    │  │
│  │  - IPortfolioRepository  - IMarketDataRepository          │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────────┐
│                    Infrastructure Layer                         │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Persistence │  │ ML Models    │  │  Optimization        │  │
│  │  - MongoDB  │  │  - PyTorch   │  │  - Gurobi            │  │
│  │  - Redis    │  │  - XGBoost   │  │  - OR-Tools          │  │
│  └─────────────┘  └──────────────┘  └──────────────────────┘  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Data ETL    │  │ Risk Mgmt    │  │  External APIs       │  │
│  │  - Airflow  │  │  - VaR/CVaR  │  │  - Yahoo Finance     │  │
│  │  - Celery   │  │  - Stress    │  │  - Alpha Vantage     │  │
│  └─────────────┘  └──────────────┘  └──────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Architectural Patterns

### 1. Clean Architecture (Hexagonal Architecture)

**Principles:**
- Dependency Inversion: Inner layers don't depend on outer layers
- Separation of Concerns: Each layer has distinct responsibilities
- Testability: Domain logic isolated from infrastructure

**Benefits:**
- Easy to test (mock infrastructure dependencies)
- Flexible to change databases, frameworks, APIs
- Domain logic remains pure and focused

### 2. Domain-Driven Design (DDD)

**Key Concepts:**

**Entities:**
- `Portfolio`: Aggregate root managing positions and allocations
- `Position`: Individual asset holdings
- `TradingSignal`: Buy/sell/hold signals

**Value Objects:**
- `RiskMetrics`: Immutable risk measurements
- `ReturnMetrics`: Performance statistics
- `Constraints`: Optimization constraints

**Aggregates:**
- Portfolio aggregate maintains consistency of all positions
- Enforces business rules (e.g., total allocation ≤ 100%)

**Domain Events:**
- `PortfolioRebalanced`: Emitted when rebalancing occurs
- `RiskThresholdExceeded`: Alert when risk exceeds tolerance
- `ModelDriftDetected`: Trigger for model retraining

**Repositories:**
- Abstract data access
- Domain layer defines interfaces
- Infrastructure layer provides implementations

### 3. Event-Driven Architecture

**Event Bus:**
- Domain events propagate through the system
- Decouples components
- Enables audit trail and event sourcing

**Examples:**
```python
# Domain event emission
portfolio.rebalance(new_allocations)
events = portfolio.get_domain_events()
# → PortfolioRebalanced event

# Event handlers
@event_handler(PortfolioRebalanced)
def send_rebalancing_notification(event):
    email_service.notify_user(event.portfolio_id)

@event_handler(RiskThresholdExceeded)
def trigger_risk_alert(event):
    alerting_service.send_alert(event)
```

---

## Domain Model

### Portfolio Aggregate

```python
Portfolio
├── portfolio_id: UUID (Identifier)
├── name: str
├── initial_capital: Decimal
├── current_value: Decimal
├── risk_tolerance: Decimal
├── positions: Dict[str, Position]
│   └── Position
│       ├── ticker: str
│       ├── quantity: Decimal
│       ├── cost_basis: Decimal
│       ├── current_price: Decimal
│       └── allocation: Decimal
├── risk_metrics: RiskMetrics
│   ├── portfolio_volatility
│   ├── var_95 / var_99
│   ├── expected_shortfall
│   └── max_drawdown
└── return_metrics: ReturnMetrics
    ├── total_return
    ├── sharpe_ratio
    ├── sortino_ratio
    └── calmar_ratio
```

### Business Rules

**Invariants (Always Enforced):**
1. Total position allocation ≤ 100%
2. Risk tolerance ∈ [0, 1]
3. Initial capital > 0
4. Position quantity ≥ 0
5. Asset prices > 0

**Domain Logic:**
- Rebalancing calculates turnover and transaction costs
- Position updates trigger portfolio value recalculation
- Risk metric updates check against tolerance thresholds

---

## Technology Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | Python 3.11+ | Main application language |
| **API** | FastAPI | REST API framework |
| **Dashboard** | Streamlit | Interactive UI |
| **Async** | Asyncio/HTTPX | Async operations |

### Data Storage

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Document DB** | MongoDB | Portfolio data, transactions |
| **Cache** | Redis | Feature store, caching |
| **Time Series** | TimescaleDB (optional) | Market data |
| **SQL** | PostgreSQL | MLflow backend |

### Machine Learning

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Classical ML** | Scikit-learn | Random Forest, ensemble |
| **Gradient Boosting** | XGBoost, LightGBM, CatBoost | Tree-based models |
| **Deep Learning** | PyTorch, TensorFlow | LSTM, neural networks |
| **Time Series** | Statsmodels | GARCH, ARIMA |
| **Experiment Tracking** | MLflow | Model registry, versioning |

### Optimization

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Commercial** | Gurobi | QP, MIQP, high performance |
| **Open Source** | OR-Tools | Alternative solver |
| **Convex** | CVXPY | Convex optimization |

### MLOps

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Monitoring** | Evidently AI | Drift detection |
| **Metrics** | Prometheus | Performance metrics |
| **Visualization** | Grafana | Dashboards |
| **Orchestration** | Apache Airflow | Workflow management |
| **Queue** | Celery + Redis | Background tasks |

### Testing

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Unit Tests** | Pytest | Unit testing |
| **Property Tests** | Hypothesis | Property-based testing |
| **Data Quality** | Great Expectations | Data validation |
| **Load Testing** | Locust | Performance testing |

---

## Data Flow

### 1. Market Data Ingestion

```
External APIs → ETL Pipeline → Data Validation → Raw Storage (MongoDB)
    ↓                ↓              ↓                    ↓
Yahoo Finance    Cleaning      Schema Check      Historical Prices
Alpha Vantage    Transform     Anomaly Detect    Company Info
FRED             Deduplicate   Quality Metrics   Macro Data
```

**ETL Pipeline (Apache Airflow):**
```python
DAG: market_data_ingestion (Daily)
├── Task 1: Fetch data from APIs
├── Task 2: Clean & validate data
├── Task 3: Store in MongoDB
├── Task 4: Update feature store (Redis)
└── Task 5: Trigger model predictions
```

### 2. Feature Engineering

```
Raw Data → Feature Engineering → Feature Store (Redis)
    ↓             ↓                      ↓
Prices        Technical Ind.        Cached Features
Volume        Fundamental           Low Latency
Macro         Risk Factors          Real-time
```

**Features:**
- **Technical**: RSI, MACD, Bollinger Bands, Moving Averages
- **Fundamental**: P/E, P/B, ROE, Debt Ratios
- **Macro**: Interest rates, Inflation, GDP growth
- **Risk**: Volatility, Beta, Correlations
- **Momentum**: Past returns, Trend indicators

### 3. ML Prediction Flow

```
Feature Store → ML Models → Predictions → Optimization
     ↓             ↓             ↓             ↓
  Features    Ensemble      Expected      Portfolio
  Redis       (RF+XGB+     Returns +      Weights
  Cache       LSTM)        Volatility
```

**Prediction Pipeline:**
1. Retrieve features from store
2. Preprocess (scaling, encoding)
3. Ensemble prediction (multiple models)
4. Uncertainty quantification
5. Store predictions for optimization

### 4. Optimization Flow

```
Predictions → Optimizer → Solution → Validation → Execution
     ↓           ↓            ↓          ↓           ↓
Expected    Gurobi/      Optimal    Business    Rebalance
Returns     OR-Tools     Weights    Rules       Portfolio
Covariance  Constraints  Sharpe     Risk Check  Transaction
```

**Optimization Steps:**
1. Formulate optimization problem
2. Add constraints (budget, sectors, etc.)
3. Solve with Gurobi/OR-Tools
4. Validate solution against business rules
5. Calculate transaction costs
6. Generate rebalancing orders

---

## ML Pipeline

### Model Architecture

#### 1. Return Prediction (Ensemble)

```python
Input Features (N) → Models → Meta-Learner → Final Prediction
                        ↓
    ┌──────────────┬──────────┬──────────┬─────────┐
    │ Random       │ XGBoost  │ LightGBM │ LSTM    │
    │ Forest       │          │          │ (PyTor  │
    │              │          │          │  ch)    │
    └──────┬───────┴────┬─────┴────┬─────┴────┬────┘
           │            │          │          │
           └────────────┴──────────┴──────────┘
                        │
                  Stacking Layer
                   (Gradient
                    Boosting)
                        │
                   Final Return
                   Prediction
```

**Training Process:**
1. **Data Split**: Time-series split (no leakage)
2. **Feature Engineering**: 50+ technical/fundamental features
3. **Base Models**: Train RF, XGBoost, LightGBM, LSTM independently
4. **Meta-Learning**: Train gradient boosting on base predictions
5. **Validation**: Walk-forward validation
6. **Model Registry**: Save to MLflow with version control

#### 2. Volatility Prediction (GARCH + LSTM)

```python
Historical Returns → GARCH Model → Volatility Forecast
                  ↓
                  LSTM Model → Volatility Forecast
                  ↓
           Ensemble Average → Final Volatility
```

**GARCH Model:**
- Captures volatility clustering
- Conditional heteroskedasticity
- Mean reversion in volatility

**LSTM Model:**
- Learns long-term dependencies
- Non-linear volatility patterns
- Regime changes

#### 3. Anomaly Detection

```python
Market Data → Isolation Forest → Anomaly Score
           ↓
           Autoencoder → Reconstruction Error
           ↓
        Threshold Check → Alert
```

**Detects:**
- Market crashes
- Flash crashes
- Regime changes
- Data quality issues

### Model Training Schedule

| Model | Frequency | Trigger |
|-------|-----------|---------|
| Return Predictor | Weekly | Scheduled + Drift |
| Volatility Model | Daily | Market close |
| Anomaly Detector | Monthly | Scheduled |
| Portfolio Optimizer | On-demand | User request |

---

## Optimization Engine

### Supported Strategies

#### 1. Markowitz Mean-Variance

**Objective:**
```
minimize: λ * w^T Σ w - μ^T w
subject to:
    Σ w_i = 1 (budget)
    w_min ≤ w_i ≤ w_max
```

**Implementation:**
- Quadratic Programming (QP)
- Gurobi: ~100ms for 50 assets
- Supports sector constraints, cardinality

#### 2. Maximum Sharpe Ratio

**Reformulation:**
```
maximize: (μ^T w - r_f) / sqrt(w^T Σ w)
→ minimize: w^T Σ w
subject to: μ^T w = 1
```

#### 3. Risk Parity

**Objective:**
```
Equal risk contribution from each asset
RC_i = w_i * (Σw)_i / (w^T Σ w) = 1/N
```

**Challenge:** Non-convex optimization
**Solution:** Sequential quadratic approximation

#### 4. CVaR Optimization

**Objective:**
```
minimize: VaR_α + 1/(1-α) * E[max(0, -R - VaR_α)]
```

**Advantages:**
- Coherent risk measure
- Captures tail risk
- Better than VaR for fat-tailed distributions

#### 5. Black-Litterman

**Combines:**
- Market equilibrium (CAPM)
- Investor views
- Bayesian updating

**Formula:**
```
E[R] = [(τΣ)^-1 + P^T Ω^-1 P]^-1 [(τΣ)^-1 Π + P^T Ω^-1 Q]
```

#### 6. Cardinality-Constrained

**Mixed-Integer Programming:**
```
Select exactly K assets from N
Binary variables: z_i ∈ {0, 1}
w_i ≤ z_i * w_max
Σ z_i = K
```

**Challenges:**
- NP-hard problem
- Gurobi uses branch-and-bound
- 1% optimality gap acceptable

---

## Risk Management

### Value at Risk (VaR)

**Three Methods:**

1. **Historical VaR**
   - Use actual historical returns
   - 95th percentile of losses
   - Simple but assumes history repeats

2. **Parametric VaR**
   - Assume normal distribution
   - VaR = μ - z_α * σ
   - Fast but may underestimate tail risk

3. **Monte Carlo VaR**
   - Simulate 10,000+ scenarios
   - Most flexible
   - Can model complex distributions

### Expected Shortfall (CVaR)

**Definition:**
Average loss in worst (1-α)% cases

**Advantages over VaR:**
- Sub-additive (coherent risk measure)
- Captures severity of tail losses
- Better for optimization

### Stress Testing

**Historical Scenarios:**
- 2008 Financial Crisis (-38% equity)
- 2020 COVID Crash (-34% equity)
- 2022 Inflation Shock (-18% equity, -13% bonds)
- 1987 Black Monday (-22% equity)
- Dot-com Bubble (-49% tech)

**Hypothetical Scenarios:**
- Interest rate shocks (+/- 200 bps)
- Recession scenario
- Inflation surge
- Currency crisis

**Implementation:**
```python
# Apply shocks to portfolio
for scenario in CRISIS_SCENARIOS:
    portfolio_loss = portfolio_value * Σ(w_i * shock_i)
    if portfolio_loss > threshold:
        trigger_alert()
```

### Drawdown Analysis

**Metrics:**
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Drawdown**: Mean of all drawdowns
- **Recovery Time**: Time to regain peak value
- **Drawdown Duration**: Length of underwater period

---

## MLOps Infrastructure

### Model Lifecycle

```
Development → Training → Validation → Deployment → Monitoring
     ↓            ↓           ↓            ↓            ↓
  Jupyter     MLflow      Walk-Forward  Production   Drift
  Notebook    Tracking    Validation    API          Detection
              Registry                  A/B Test     Auto-Retrain
```

### Drift Detection

**Data Drift Tests:**
1. **Kolmogorov-Smirnov**: Distribution changes
2. **Population Stability Index**: Feature shifts
3. **Wasserstein Distance**: Distribution distance

**Concept Drift:**
- Performance degradation monitoring
- Prediction distribution shifts
- Directional accuracy decline

**Actions on Drift:**
```python
if drift_detector.detect():
    if drift_severity > CRITICAL_THRESHOLD:
        # Immediate retraining
        trigger_retraining()
        notify_team()
    elif drift_severity > WARNING_THRESHOLD:
        # Schedule retraining
        schedule_retraining(delay=1_day)
    
    # Log drift event
    mlflow.log_metric("drift_score", drift_score)
```

### A/B Testing Framework

**Compare Strategies:**
```python
# Split portfolio into test groups
group_a = use_strategy("markowitz")
group_b = use_strategy("risk_parity")

# Track performance
for t in range(90_days):
    performance_a = evaluate(group_a)
    performance_b = evaluate(group_b)
    
    # Statistical test
    if is_significant(performance_a, performance_b):
        promote_winner()
```

### Model Monitoring

**Metrics Tracked:**
- Prediction accuracy (RMSE, MAE, R²)
- Directional accuracy
- Sharpe ratio of portfolio
- Transaction costs
- Rebalancing frequency
- Feature importance drift

**Alerts:**
- Email/Slack notifications
- PagerDuty for critical issues
- Grafana dashboard updates

---

## Scalability & Performance

### Performance Optimizations

**1. Caching Strategy:**
```python
@redis_cache(ttl=3600)
def get_features(ticker):
    # Expensive feature computation
    return features
```

**2. Batch Processing:**
- Process multiple portfolios in parallel
- Vectorized numpy operations
- GPU acceleration for neural networks

**3. Async Operations:**
```python
async def fetch_all_prices(tickers):
    tasks = [fetch_price(ticker) for ticker in tickers]
    return await asyncio.gather(*tasks)
```

**4. Database Indexing:**
```javascript
// MongoDB indexes
db.portfolios.createIndex({ "owner_id": 1, "created_at": -1 })
db.positions.createIndex({ "portfolio_id": 1, "ticker": 1 })
db.market_data.createIndex({ "ticker": 1, "date": -1 })
```

### Horizontal Scaling

**Stateless Services:**
- API servers: Load balanced
- Celery workers: Add more workers
- ML inference: Deploy multiple instances

**Load Balancing:**
```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: robo-advisor-api
spec:
  replicas: 5
  selector:
    matchLabels:
      app: robo-advisor-api
```

### Performance Benchmarks

| Operation | Time | Throughput |
|-----------|------|------------|
| Markowitz Optimization (50 assets) | 100ms | 10 req/s |
| CVaR Optimization (100 assets) | 500ms | 2 req/s |
| VaR Calculation | 50ms | 20 req/s |
| ML Prediction | 30ms | 33 req/s |
| Stress Test (5 scenarios) | 200ms | 5 req/s |

---

## Security Considerations

1. **Authentication & Authorization:**
   - JWT tokens
   - Role-based access control (RBAC)
   - API key management

2. **Data Encryption:**
   - TLS/SSL for data in transit
   - Encryption at rest (MongoDB)
   - Secure credential storage (HashiCorp Vault)

3. **Input Validation:**
   - Pydantic models
   - SQL injection prevention
   - Rate limiting

4. **Audit Logging:**
   - All portfolio changes logged
   - Immutable event store
   - Compliance reporting

---

## Deployment Architecture

### Production Setup

```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancer (Nginx)                │
└─────────────────────────┬───────────────────────────────┘
                          │
          ┌───────────────┴──────────────┐
          │                              │
┌─────────▼────────┐          ┌──────────▼──────────┐
│   API Server 1   │          │   API Server 2-N    │
│   (Container)    │          │   (Containers)      │
└─────────┬────────┘          └──────────┬──────────┘
          │                              │
          └───────────────┬──────────────┘
                          │
          ┌───────────────┴──────────────┐
          │                              │
┌─────────▼────────┐          ┌──────────▼──────────┐
│   MongoDB        │          │   Redis Cluster     │
│   Replica Set    │          │   (Cache + Queue)   │
└──────────────────┘          └─────────────────────┘
```

**Infrastructure as Code:**
- Terraform for cloud resources
- Kubernetes for orchestration
- Helm charts for deployment
- CI/CD with GitHub Actions

---

This architecture provides a solid foundation for an enterprise-grade robo-advisor with:
- ✅ Clean, maintainable code
- ✅ Scalable infrastructure
- ✅ Production-ready MLOps
- ✅ Comprehensive risk management
- ✅ High performance optimization
