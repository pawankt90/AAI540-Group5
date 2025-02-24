from sagemaker.image_uris import retrieve
from utils import load_config
import boto3
import sagemaker
import json
import time
import os
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
import numpy as np
from pyathena import connect
import pandas as pd

# Initialize AWS Session
session = boto3.session.Session()
sagemaker_session = sagemaker.Session()
config = load_config()
bucket = config['bucket']
s3_bucket = config['bucket']
prefix = "flight-delay-prediction-xgboost"
role = load_config()["sagemaker_execution_role"]
region = load_config()["region"]


def load_trained_model(s3_model_path):
    """Downloads and extracts the trained XGBoost model from S3."""
    s3_client = boto3.client('s3')
    local_model_path = "xgboost_model.tar.gz"

    # Extract bucket name and key from S3 path
    bucket_name = s3_model_path.split('/')[2]
    key = "/".join(s3_model_path.split('/')[3:])

    print(f"Downloading model from {s3_model_path} to {local_model_path}...")
    s3_client.download_file(bucket_name, key, local_model_path)

    # Unzip model
    os.system(f"tar -xvzf {local_model_path}")
    print("✅ Model extracted successfully.")

    return "xgboost-model"  # The model file name inside .tar.gz


# Deploy Model to SageMaker Endpoint
def deploy_model(s3_model_path):

    """Deploys the trained XGBoost model as a real-time SageMaker endpoint."""
    model_name = "xgboost-flight-delay-model"
    endpoint_name = "flight-delay-xgboost-endpoint"

    # Check and delete existing endpoint
    delete_existing_model(model_name)
    delete_existing_endpoint(endpoint_name)

    # Create a SageMaker Model from the trained model artifact
    xgb_model = sagemaker.model.Model(
        name=model_name,
        image_uri=sagemaker.image_uris.retrieve("xgboost", region, "1.5-1"),
        model_data=s3_model_path,
        role=role,
        sagemaker_session=sagemaker_session
    )

    from sagemaker.model_monitor import DataCaptureConfig
    # :white_check_mark: Enable data capture for monitoring
    data_capture_config = DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,  # Capture 100% of inference data
        destination_s3_uri=f"s3://{bucket}/data_capture/",
        capture_options=["Input", "Output"]
    )

    predictor = xgb_model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.xlarge",
        endpoint_name=endpoint_name,
        serializer=CSVSerializer(),
        deserializer=JSONDeserializer(),
        data_capture_config=data_capture_config  # :white_check_mark: Added data capture config
    )

    print(f"✅ Model deployed at endpoint: {endpoint_name}")
    return predictor, endpoint_name

def delete_existing_model(model_name):
    """Deletes the SageMaker model if it exists."""
    sm_client = boto3.client("sagemaker")
    try:
        sm_client.describe_model(ModelName=model_name)
        print(f"⚠️ Model {model_name} exists. Deleting...")
        sm_client.delete_model(ModelName=model_name)
        time.sleep(5)  # Wait for deletion to complete
        print(f"✅ Model {model_name} deleted.")
    except sm_client.exceptions.ClientError:
        print(f"✅ Model {model_name} does not exist. Proceeding with deployment.")


# Setup Batch Transform for Large Inference Jobs
def batch_transform(s3_input_path,s3_model_path):
    """Creates a Batch Transform job for large-scale inference."""
    batch_output_s3 = f"s3://{bucket}/{prefix}/batch-output"
    transformer = sagemaker.transformer.Transformer(
        model_name="xgboost-flight-delay-model",
        instance_count=1,
        instance_type="ml.m5.xlarge",
        output_path=batch_output_s3,
        sagemaker_session=sagemaker_session
    )

    transformer.transform(
        data=s3_input_path,
        content_type="text/csv",
        split_type="Line"
    )

    print("✅ Batch Transform job submitted. Waiting for completion...")
    transformer.wait()
    print(f"✅ Batch Transform results stored at: {batch_output_s3}")
    endpoint_name_batch_transform = "flight-delay-xgboost-endpoint-with-batch-transform"
    model_name = "xgboost-flight-delay-model"
    xgb_model = sagemaker.model.Model(
        name=model_name,
        image_uri=sagemaker.image_uris.retrieve("xgboost", region, "1.5-1"),
        model_data=s3_model_path,
        role=role,
        sagemaker_session=sagemaker_session
    )
    delete_existing_endpoint(endpoint_name_batch_transform)
    from sagemaker.model_monitor import DataCaptureConfig
    # :white_check_mark: Enable data capture for monitoring
    data_capture_config = DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,  # Capture 100% of inference data
        destination_s3_uri=f"s3://{bucket}/data_capture/",
        capture_options=["Input", "Output"]
    )

    predictor_batch_request = xgb_model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.xlarge",
        endpoint_name=endpoint_name_batch_transform,
        serializer=CSVSerializer(),  # <--- important
        deserializer=JSONDeserializer(),  # or StringDeserializer() depending on your output,
        data_capture_config = data_capture_config  # :white_check_mark: Added data capture config
    )
    return batch_output_s3


# Cleanup SageMaker Endpoint
def cleanup_endpoint(endpoint_name):
    """Deletes the SageMaker endpoint to free resources."""
    predictor = sagemaker.predictor.Predictor(endpoint_name, sagemaker_session)
    predictor.delete_endpoint()
    print(f"✅ Endpoint {endpoint_name} deleted.")

def get_feature_store_data(feature_store_table, s3_staging_dir, region):
    conn = connect(s3_staging_dir=s3_staging_dir, region_name=region)
    query = f'SELECT * FROM "sagemaker_featurestore"."{feature_store_table}";'
    return pd.read_sql(query, conn)

def get_feature_store_table_name(feature_group_name):
    print(f"⏳ Waiting for the Feature Group '{feature_group_name}' to be available in Glue...")
    sagemaker_client = boto3.client("sagemaker", region_name=region)
    # Wait for Feature Group to be created
    while True:
        response = sagemaker_client.describe_feature_group(FeatureGroupName=feature_group_name)
        status = response["FeatureGroupStatus"]

        if status == "Created":
            print(f"✅ Feature Group '{feature_group_name}' is now active!")
            break

        print(f"⏳ Current status: {status}, retrying in 5 seconds...")
        time.sleep(5)

    # Retrieve Glue Table Name from Offline Store Config
    try:
        table_name = response["OfflineStoreConfig"]["DataCatalogConfig"]["TableName"]
        print(f"✅ Feature Store table registered in Glue for '{feature_group_name}': {table_name}")
        return table_name
    except KeyError:
        print(f"❌ Error: Offline Store is not properly configured for '{feature_group_name}'.")
        return None


# Main Execution
def get_batch_input(df):
    small_test_set = df.drop('on_time', axis=1).sample(500)

    small_test_csv_path = "small_test.csv"
    small_test_set.to_csv(small_test_csv_path, index=False, header=False)

    # Upload smaller dataset to S3
    small_test_s3_path = sagemaker_session.upload_data(
        path=small_test_csv_path,
        bucket=bucket,
        key_prefix=prefix + "/small_test"
    )

    return small_test_s3_path

def prepare_data(df):
    df.drop('record_id', axis=1, inplace=True)
    df.drop('write_time', axis=1, inplace=True)
    df.drop('api_invocation_time', axis=1, inplace=True)
    df.drop('is_deleted', axis=1, inplace=True)
    print(df.head())
    from sklearn.preprocessing import LabelEncoder

    for col in ['carrier', 'airport']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    df.fillna(0, inplace=True)
    return df


def test_model_prediction(endpoint_name, test_data):
    """Tests the deployed model by making a prediction on sample data."""
    predictor = sagemaker.predictor.Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sagemaker_session,
        serializer=CSVSerializer(),
        deserializer=JSONDeserializer()
    )

    # Convert test data to CSV format
    test_data_csv = ",".join(map(str, test_data))
    response = predictor.predict(test_data_csv)

    print(f"✅ Model Prediction: {response}")
    return response

# Check and Delete Existing Endpoint
# Check and Delete Existing Endpoint and Configuration
def delete_existing_endpoint(endpoint_name):
    """Deletes the SageMaker endpoint and its configuration if they exist."""
    sm_client = boto3.client("sagemaker")

    try:
        # Check if endpoint exists
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        print(f"⚠️ Endpoint {endpoint_name} exists. Deleting...")
        sm_client.delete_endpoint(EndpointName=endpoint_name)

        # Wait for the deletion to complete
        while True:
            try:
                sm_client.describe_endpoint(EndpointName=endpoint_name)
                time.sleep(5)
            except sm_client.exceptions.ClientError:
                print(f"✅ Endpoint {endpoint_name} deleted.")
                break

        # Check if endpoint configuration exists and delete it
        print(f"⚠️ Deleting endpoint configuration for {endpoint_name}...")
        sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print(f"✅ Endpoint configuration {endpoint_name} deleted.")

    except sm_client.exceptions.ClientError:
        print(f"✅ Endpoint {endpoint_name} does not exist. Proceeding with deployment.")

def store_model(model_artifact_path, s3_bucket):
    s3_client = boto3.client('s3')

    local_model_path = "xgboost_model.tar.gz"

    # Extract bucket name and key from S3 path
    bucket_name = model_artifact_path.split('/')[2]
    key = "/".join(model_artifact_path.split('/')[3:])
    print(f"Downloading model from {model_artifact_path} to {local_model_path}...")
    s3_client.download_file(bucket_name, key, local_model_path)

    s3_model_path = f"xgboost_models/xgboost_model.tar.gz"
    s3_client.upload_file(local_model_path, s3_bucket, s3_model_path)
    print(f"✅ Model stored at s3://{s3_bucket}/{s3_model_path}")
    return f"s3://{s3_bucket}/{s3_model_path}"

def get_approved_model_path():
    ssm_client = boto3.client("ssm")
    # ✅ Retrieve the correct model package ARN from SSM
    response = ssm_client.get_parameter(Name="/pipeline/current_model_package_arn")
    model_package_arn = response["Parameter"]["Value"]
    sm_client = boto3.client("sagemaker")
    sm_client.update_model_package(
        ModelPackageArn=model_package_arn,
        ModelApprovalStatus="Approved"
    )
    model_info = sm_client.describe_model_package(ModelPackageName=model_package_arn)
    s3_model_path = model_info["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]

    return s3_model_path


def main():

    config = load_config()
    s3_model_path = config["xgboost_model"]
    config = load_config()
    #s3_model_path = config['xgboost_model']  # Replace with actual path
    s3_staging_dir = config["s3_staging_dir"]
    # Load model and deploy to endpoint
    #model_file = load_trained_model(s3_model_path)
    s3_model_path = get_approved_model_path()
    xgboost_deployed_path = store_model(s3_model_path, s3_bucket)
    predictor, endpoint_name = deploy_model(xgboost_deployed_path)

    # Run Batch Transform
    dev_feature_store_table = get_feature_store_table_name(config["dev_fg_name"])
    df = get_feature_store_data(dev_feature_store_table, s3_staging_dir, region)
    batch_input_s3 = get_batch_input(prepare_data(df))  # Replace with actual batch input
    batch_transform(batch_input_s3,s3_model_path)

    # Test model with a sample input
    sample_input = np.random.rand(20)  # Replace with a real feature vector
    test_model_prediction(endpoint_name, sample_input)
    delete_existing_model("xgboost-flight-delay-model")
    delete_existing_endpoint("flight-delay-xgboost-endpoint")
    delete_existing_endpoint("flight-delay-xgboost-endpoint-with-batch-transform")

    #print(f"✅ Deployment complete. Endpoint: {endpoint_name}")



if __name__ == "__main__":
    main()
