FROM python:3.11-slim
WORKDIR /app

# Copy your Python dependencies
COPY requirements.txt .

# Copy local wheels folder
COPY wheels/ ./wheels

# Install dependencies from local wheels only (no internet)
RUN pip install --no-index --find-links=wheels -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY .env .env

# Create necessary directories
RUN mkdir -p mlruns data logs

EXPOSE 8000

CMD ["uvicorn", "src.presentation.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
