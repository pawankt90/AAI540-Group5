# utils.py
import boto3
import sagemaker
import logging
import yaml
from urllib.parse import urlparse
import os

def load_config(config_path="config.yaml"):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

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