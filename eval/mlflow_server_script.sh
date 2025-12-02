#!/bin/bash

#set the DB_URI environment variable correctly

# S3 artifact location
ARTIFACT_ROOT="s3://wildtts-metrics-eval-artifacts/mlflow"

# Start MLflow tracking server
mlflow server \
  --backend-store-uri "$DB_URI" \
  --default-artifact-root "$ARTIFACT_ROOT" \
  --host 0.0.0.0 \
  --port 5000