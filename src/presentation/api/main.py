from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.presentation.api.routes import portfolio, ml, health

app = FastAPI(
    title="Robo-Advisor API",
    description="Portfolio Optimization with ML",
    version="1.0.0"
)

# CORS - MÉTHODE CORRECTE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["portfolio"])
app.include_router(ml.router, prefix="/api/v1/ml", tags=["machine-learning"])

@app.get("/")
async def root():
    return {
        "message": "🚀 Robo-Advisor API",
        "version": "1.0.0",
        "docs": "/docs"
    }
