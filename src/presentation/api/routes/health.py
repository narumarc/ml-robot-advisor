"""
Health Check Endpoints
"""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str = "1.0.0"


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns status of the API.
    """
    return HealthResponse(
        status="healthy",
        service="robo-advisor-api",
        version="1.0.0"
    )


@router.get("/ping")
async def ping():
    """Simple ping endpoint"""
    return {"message": "pong"}
