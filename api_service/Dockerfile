FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY api_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model files
COPY api_service/main.py .
COPY numeric_similarity_model.py .
COPY api_service/model.pkl .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]