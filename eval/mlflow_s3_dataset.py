#!/usr/bin/env python3
"""
Simple script to create an MLFlow dataset from a folder and store it in AWS S3.

Requirements for S3 and MLFlow integration:
1. AWS S3 bucket with appropriate permissions (read/write access)
2. AWS credentials configured (via ~/.aws/credentials or environment variables):
   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_DEFAULT_REGION (optional)
3. MLFlow tracking server configured to use S3 for artifact storage:
   - Set MLFLOW_TRACKING_URI environment variable or use mlflow.set_tracking_uri()
   - Tracking server must have mlflow.s3.artifact.root configured (e.g., s3://your-bucket/mlflow-artifacts)
4. Python packages:
   - mlflow
   - boto3 (for S3 access)
   - pandas (if logging dataframes)

Note: Ensure IAM user/role has s3:PutObject, s3:GetObject, s3:ListBucket permissions
"""

import os
import argparse
import subprocess
import mlflow
from pathlib import Path
from typing import Optional, Dict, Any
import json


def create_and_log_dataset(
    folder_path: str,
    dataset_name: str,
    experiment_name: str,
    metadata: Dict[str, Any],
    description: Optional[str] = None,
    run_id: Optional[str] = None,
):
    """
    Create an MLFlow dataset from a folder and log it to S3.
    
    Args:
        folder_path: Path to the folder containing dataset files
        dataset_name: Name for the dataset in MLFlow
        experiment_name: MLFlow experiment name (default: "default")
        metadata: Optional metadata dictionary to attach to the dataset
        description: Optional description of the dataset
    """
    
    # Set MLFlow tracking URI (can also be set via environment variable)
    # Example: mlflow.set_tracking_uri("http://localhost:5000")
    # Must start the server first with DB abd artifact store configured

    mlflow.set_tracking_uri("http://localhost:5000")
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
    # Start MLFlow run
    with mlflow.start_run(run_name=f"dataset_{dataset_name}", run_id=run_id) as run:
        
        # Log folder as artifacts
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        print(f"Logging dataset from: {folder_path}")
        mlflow.log_artifacts(str(folder_path), artifact_path=dataset_name)
        
        # Log metadata as parameters
        if metadata:
            for key, value in metadata.items():
                mlflow.log_param(key, value)
        
        # Log description as a tag
        if description:
            mlflow.set_tag("description", description)
        
        # Log basic dataset info
        mlflow.set_tag("dataset_name", dataset_name)
        mlflow.set_tag("source_path", str(folder_path))
        
        # Get run info
        run = mlflow.active_run()
        print(f"\nDataset logged successfully!")
        print(f"Run ID: {run.info.run_id}")
        print(f"Experiment ID: {run.info.experiment_id}")
        print(f"Artifact URI: {run.info.artifact_uri}")
        
        return run.info.run_id
def log_metrics_from_json(run_id: str, json_path: Path):
    """Log metrics from a JSON file to an existing MLFlow run."""


    if not json_path.exists():
        raise ValueError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        metrics = json.load(f)
    
    with mlflow.start_run(run_id=run_id):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
            elif isinstance(value, dict) and value.get("mean") is not None:
                mlflow.log_metric(f"{key}_mean", value["mean"])
            else:
                print(f"Skipping metric: {key}={value}")

def main():
    """Example usage"""
    parser = argparse.ArgumentParser(description='Create an MLFlow dataset from a folder and store it in AWS S3.')
    parser.add_argument('--folder_path', type=Path, help='Path to the folder containing dataset files')
    parser.add_argument('--dataset_name', type=str, help='Name for the dataset in MLFlow')
    parser.add_argument('--experiment_name', type=str, help='MLFlow experiment name', default="WildTTS")
    parser.add_argument('--description', type=str, help='Description of the dataset', default=None)
    
    args = parser.parse_args()
    
    # Get current commit ID
    try:
        commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(__file__)).decode('ascii').strip()
    except:
        commit_id = "unknown"
    
    # Optional metadata
    metadata = {
        "commit_id": commit_id,
    }
    
    # Create and log dataset
    try:
        run_id = create_and_log_dataset(
            folder_path=args.folder_path,
            dataset_name=args.dataset_name,
            experiment_name=args.experiment_name,
            metadata=metadata,
            description=args.description
        )
        print(f"\nDataset stored with run ID: {run_id}")

        # if the comprehensive evaluation metrics are available, they can be logged here under the same run
        eval_folder_name = args.folder_path.name + "_results"
        eval_folder_path = args.folder_path.parent / eval_folder_name
        if eval_folder_path.exists():
            print(f"\nLogging comprehensive evaluation metrics from: {eval_folder_path}")
            create_and_log_dataset(
                folder_path=eval_folder_path,
                dataset_name=args.dataset_name + "_results",
                experiment_name=args.experiment_name,
                metadata=metadata,
                description="Comprehensive evaluation metrics",
                run_id=run_id
            )

            if (eval_folder_path/"statistics.json").exists():
                print(f"\nLogging statistics.json from: {eval_folder_path/'statistics.json'}")
                log_metrics_from_json(run_id, eval_folder_path/"statistics.json")
                
            print(f"\nComprehensive evaluation metrics logged under run ID: {run_id}")

        
    except Exception as e:
        print(f"Error: {e}")



if __name__ == "__main__":
    main()
