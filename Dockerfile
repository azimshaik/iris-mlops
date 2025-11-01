# Dockerfile
FROM python:3.9-slim

# Install dependencies
RUN pip install --no-cache-dir \
    scikit-learn==1.3.0 \
    pandas==2.0.3 \
    joblib==1.3.2 \
    google-cloud-bigquery==3.10.0 \
    google-cloud-storage==2.10.0 \
    db-dtypes

# Copy the training script into the container
COPY train.py /app/train.py

# Set the entrypoint
WORKDIR /app
ENTRYPOINT ["python", "train.py"]