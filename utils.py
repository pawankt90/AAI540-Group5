# utils.py
import boto3
import sagemaker
import logging
import yaml
from urllib.parse import urlparse
import os

def load_config(config_path="config.yaml"):
    """Load YAML configuration file."""
    config = {'region': 'us-east-1',
     'sagemaker_execution_role': 'arn:aws:iam::701586278703:role/service-role/AmazonSageMaker-ExecutionRole-20250214T203663',
     'ecr_repository': 'flight-delay-prediction-xgboost',
     'sagemaker_endpoint_name': 'flight-delay-xgboost-endpoint-single-request',
     'bucket': 'sagemaker-us-east-1-701586278703',
     's3_csv_private_path': 's3://sagemaker-us-east-1-701586278703/airline-delay-cause/csv/',
     'athena_database': 'db_airline_delay_cause', 'database_name': 'db_airline_delay_cause',
     'raw_table_name': 'airline_delay_cause_csv_raw',
     's3_staging_dir': 's3://sagemaker-us-east-1-701586278703/athena/staging', 'dev_table_name': 'development_data',
     'prod_table_name': 'production_data', 's3_prefix': 'flight-delay-prediction-xgboost',
     'feature_store_s3_dev': 's3://sagemaker-us-east-1-701586278703/feature-store/dev/',
     'feature_store_s3_prod': 's3://sagemaker-us-east-1-701586278703/feature-store/prod/',
     'dev_s3_uri': 's3://sagemaker-us-east-1-701586278703/feature-store/dev/',
     'prod_s3_uri': 's3://sagemaker-us-east-1-701586278703/feature-store/prod/',
     'dev_fg_name': 'airline_delay_features_dev', 'prod_fg_name': 'airline_delay_features_prod',
     'baseline_model_metrics_json': 's3://sagemaker-us-east-1-701586278703/models/model_metrics.json',
     'baseline_model_pkl': 's3://sagemaker-us-east-1-701586278703/models/linear_regression_model.pkl',
     'xgboost_model': 's3://sagemaker-us-east-1-701586278703/xgboost_models/xgboost_model.tar.gz',
     'xgboost_model_metrics_json': 's3://sagemaker-us-east-1-701586278703/xgboost_models/model_metrics.json',
     'training': {'num_round': 100, 'max_depth': 5, 'eta': 0.2, 'subsample': 0.8, 'eval_metric': 'auc'},
     'monitoring': {'instance_type': 'ml.m5.xlarge', 'instance_count': 1, 'max_runtime_in_seconds': 3600,
                    'schedule': 'hourly'}}
    return config
    # with open(config_path, "r") as f:
    #     config=yaml.safe_load(f)
    #     print(config)
    #     return yaml.safe_load(f)

def init_aws_session(region=None):
    """Initialize and return a boto3 session and SageMaker session."""
    session = boto3.session.Session(region_name=region)
    sagemaker_session = sagemaker.Session(session)
    return session, sagemaker_session

def get_default_bucket(sagemaker_session):
    """Return the default SageMaker bucket."""
    return sagemaker_session.default_bucket()

def setup_logger(name=__name__):
    """Set up and return a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def copy_to_s3(src_path, dest_path):
    config = load_config()
    session, sagemaker_session = init_aws_session(config['region'])
    s3 = boto3.client("s3")
    # Parse the bucket name and key
    parsed_url = urlparse(dest_path)
    bucket_name = parsed_url.netloc
    key_prefix = parsed_url.path.lstrip("/")  # Ensure no leading slash

    # Generate the full S3 key by appending the filename
    object_key = os.path.join(key_prefix, os.path.basename(src_path))

    # Upload file
    s3.upload_file(src_path, bucket_name, object_key)

    print(f"Uploaded {src_path} to s3://{bucket_name}/{object_key}")