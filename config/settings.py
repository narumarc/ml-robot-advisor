"""
Settings avec Pydantic
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # App
    APP_NAME: str = "Robo-Advisor"
    DEBUG: bool = False
    
    # MongoDB
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "robo_advisor"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_TTL: int = 3600  # 1 hour
    
    # MLflow
    MLFLOW_TRACKING_URI: str = "file:./mlruns"
    MLFLOW_EXPERIMENT_NAME: str = "robo-advisor"
    
    # Data
    DATA_SOURCE: str = "stooq"  # 'stooq', 'alphavantage', 'csv'
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    
    # ML
    DEFAULT_MODEL_TYPE: str = "random_forest"
    TARGET_HORIZON_DAYS: int = 20
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()