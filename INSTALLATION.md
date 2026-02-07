# 📦 Guide d'Installation Complet

## Prérequis

### Système
- **OS**: Linux, macOS, ou Windows (avec WSL2)
- **Python**: 3.10 ou 3.11
- **RAM**: 8 GB minimum
- **Disk**: 5 GB d'espace libre

### Logiciels
- Docker & Docker Compose
- Git
- Python 3.10+
- pip & virtualenv

---

## Installation Pas à Pas

### 1. Cloner/Extraire le Projet

```bash
# Si depuis Git
git clone https://github.com/votre-repo/robo-advisor-project.git
cd robo-advisor-project

# Ou extraire le ZIP
unzip robo-advisor-project.zip
cd robo-advisor-project
```

### 2. Environnement Virtuel Python

```bash
# Créer environnement virtuel
python3.10 -m venv venv

# Activer
# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# Vérifier
which python  # Doit pointer vers venv/bin/python
python --version  # Doit être 3.10+
```

### 3. Installer les Dépendances

```bash
# Upgrade pip
pip install --upgrade pip

# Installer requirements
pip install -r requirements.txt

# Vérifier installation
pip list | grep xgboost
pip list | grep fastapi
```

**Note**: Si erreur avec Gurobi, voir section Gurobi ci-dessous.

### 4. Configuration

```bash
# Copier exemple de config
cp .env.example .env

# Éditer .env
nano .env  # ou vim, code, etc.
```

**Variables importantes dans .env:**
```bash
# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=robo_advisor

# Redis
REDIS_URL=redis://localhost:6379

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# API Keys (optionnel)
ALPHA_VANTAGE_API_KEY=your_key_here
```

### 5. Docker Infrastructure

```bash
# Vérifier Docker
docker --version
docker-compose --version

# Démarrer les services
docker-compose up -d

# Vérifier les services
docker-compose ps

# Logs
docker-compose logs -f mongodb
docker-compose logs -f redis
```

**Services lancés:**
- MongoDB: `localhost:27017`
- Redis: `localhost:6379`
- MLflow: `http://localhost:5000`
- PostgreSQL (MLflow backend): `localhost:5432`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

### 6. Initialiser la Base de Données

```bash
# MongoDB indexes
python scripts/init_db.py

# Vérifier
mongo robo_advisor --eval "db.getCollectionNames()"
```

### 7. Télécharger Données de Test (Optionnel)

```bash
# Créer répertoires
mkdir -p data/{train,test,reference,production}

# Télécharger données via script
python scripts/download_sample_data.py
```

---

## Installation Gurobi (Optionnel mais Recommandé)

### Option 1: Licence Académique (Gratuite)

```bash
# 1. Créer compte sur gurobi.com
# 2. Obtenir licence académique gratuite
# 3. Télécharger Gurobi

# 4. Installer
pip install gurobipy

# 5. Activer licence
grbgetkey YOUR_LICENSE_KEY

# 6. Vérifier
python -c "import gurobipy; print(gurobipy.gurobi.version())"
```

### Option 2: Trial (15 jours)

```bash
pip install gurobipy
# Fonctionne sans licence pour 15 jours
```

### Option 3: Alternative (CVXPY)

Si pas de licence Gurobi:
```bash
pip install cvxpy
# Modifie portfolio_optimizer.py pour utiliser CVXPY
```

---

## Vérification de l'Installation

### Test Complet

```bash
# Script de vérification
python scripts/verify_installation.py
```

**Ce script vérifie:**
- ✅ Python version
- ✅ Packages installés
- ✅ MongoDB connection
- ✅ Redis connection
- ✅ MLflow accessible
- ✅ Gurobi disponible

### Tests Unitaires

```bash
# Lancer tests
pytest tests/unit/ -v

# Devrait afficher:
# ✅ tests/unit/test_domain_entities.py PASSED
# ✅ tests/unit/test_optimization.py PASSED
```

---

## Problèmes Courants

### 1. ModuleNotFoundError

```bash
# Si "No module named 'src'"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Ou ajouter à ~/.bashrc
echo 'export PYTHONPATH="${PYTHONPATH}:/path/to/robo-advisor-project"' >> ~/.bashrc
```

### 2. MongoDB Connection Error

```bash
# Vérifier que Docker est lancé
docker-compose ps

# Redémarrer MongoDB
docker-compose restart mongodb

# Vérifier logs
docker-compose logs mongodb
```

### 3. Redis Connection Error

```bash
# Vérifier Redis
docker-compose ps redis

# Test connection
redis-cli -h localhost -p 6379 ping
# Devrait retourner: PONG
```

### 4. Gurobi License Error

```bash
# Vérifier licence
gurobi_cl --version

# Réactiver
grbgetkey YOUR_LICENSE_KEY

# Alternative: utiliser CVXPY
pip install cvxpy
```

### 5. Port Already in Use

```bash
# Trouver processus
lsof -i :8000  # FastAPI
lsof -i :5000  # MLflow
lsof -i :27017 # MongoDB

# Tuer processus
kill -9 PID
```

---

## Configuration Avancée

### MLflow avec S3 (Production)

```bash
# .env
MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
MLFLOW_ARTIFACT_ROOT=s3://your-bucket/mlflow
```

### MongoDB Replica Set (Production)

```yaml
# docker-compose.prod.yml
services:
  mongodb:
    image: mongo:7.0
    command: mongod --replSet rs0
    # ... configuration replica set
```

### Airflow Setup

```bash
# Installer Airflow
pip install apache-airflow

# Initialiser
airflow db init

# Créer user
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Copier DAGs
cp mlops/airflow/dags/*.py ~/airflow/dags/

# Démarrer
airflow webserver --port 8080 &
airflow scheduler &
```

---

## Désinstallation

### Arrêter Services

```bash
# Arrêter Docker
docker-compose down

# Supprimer volumes (attention: efface données)
docker-compose down -v
```

### Supprimer Environnement

```bash
# Désactiver venv
deactivate

# Supprimer
rm -rf venv/
```

### Nettoyer Données

```bash
# Supprimer données
rm -rf data/
rm -rf models/
rm -rf reports/
```

---

## Next Steps

Une fois installé:

1. **Lire README_FINAL.md** - Vue d'ensemble
2. **Consulter docs/** - Documentation détaillée
3. **Exécuter scripts/demo_complete.py** - Démo complète
4. **Lancer mlops/training/train_all_models.py** - Entraîner modèles
5. **Démarrer src/presentation/api/main.py** - Lancer API

---

## Support

En cas de problème:
1. Vérifier les logs Docker: `docker-compose logs`
2. Consulter `/docs/TROUBLESHOOTING.md`
3. Vérifier requirements: `pip list`

---

**Installation réussie ? Passe à README_FINAL.md pour commencer ! 🚀**
