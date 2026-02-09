"""
Configuration settings for the Robo Advisor application.
Uses Pydantic for settings management and validation.
"""
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    env: str = Field(default="development", description="Environment: development, staging, production")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    
    # Database
    mongodb_uri: str = Field(default="mongodb://localhost:27017")
    mongodb_database: str = Field(default="robo_advisor")
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    
    # API Keys
    alpha_vantage_api_key: Optional[str] = Field(default=None)
    financial_data_api_key: Optional[str] = Field(default=None)
    
    # Gurobi
    gurobi_home: Optional[str] = Field(default=None)
    grb_license_file: Optional[str] = Field(default=None)
    
    # MLflow
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_experiment_name: str = Field(default="robo_advisor")
    
    # Risk Management
    max_position_size: float = Field(default=0.15, ge=0.0, le=1.0)
    max_sector_exposure: float = Field(default=0.30, ge=0.0, le=1.0)
    min_diversification: int = Field(default=10, ge=1)
    
    # Optimization
    optimization_solver: str = Field(default="GUROBI")
    optimization_timeout: int = Field(default=300)
    
    # Rebalancing
    rebalance_threshold: float = Field(default=0.05)
    transaction_cost: float = Field(default=0.001)
    
    # Model Monitoring
    drift_detection_threshold: float = Field(default=0.1)
    retraining_threshold: float = Field(default=0.15)
    model_performance_check_interval: str = Field(default="daily")
    
    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    
    @validator("optimization_solver")
    def validate_solver(cls, v):
        """Validate optimization solver choice."""
        allowed = ["GUROBI","HIGHS", "ORTOOLS", "CVXPY"]
        if v.upper() not in allowed:
            raise ValueError(f"Solver must be one of {allowed}")
        return v.upper()
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v.upper()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
