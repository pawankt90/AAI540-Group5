import boto3
import sagemaker
from pyathena import connect
import pandas as pd
import numpy as np
import time
import json
import joblib
from sagemaker.image_uris import retrieve
from sagemaker.inputs import TrainingInput
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
from utils import load_config
import os

# Initialize AWS Session
session = boto3.session.Session()
config = load_config()
bucket = config['bucket']
sagemaker_session = sagemaker.Session()
prefix = "flight-delay-prediction-xgboost"
role = load_config()["sagemaker_execution_role"]
region = load_config()["region"]


# Function to Retrieve Feature Store Data
def get_feature_store_data(feature_store_table, s3_staging_dir, region):
    conn = connect(s3_staging_dir=s3_staging_dir, region_name=region)
    query = f'SELECT * FROM "sagemaker_featurestore"."{feature_store_table}";'
    return pd.read_sql(query, conn)


# Function to Prepare Data for Training
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
    features = [
        "arr_flights", "arr_del15", "carrier_ct", "weather_ct", "nas_ct",
        "security_ct", "late_aircraft_ct", "arr_cancelled", "arr_diverted",
        "arr_delay", "carrier_delay", "weather_delay", "nas_delay",
        "security_delay", "late_aircraft_delay", "delay_rate"
    ]
    X = df.drop('on_time', axis=1)
    #X = df[features]
    y = df['on_time']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    feature_order = X_train.columns.tolist()
    # train_combined = pd.concat([y_train, X_train[feature_order]], axis=1)
    # val_combined = pd.concat([y_val, X_val[feature_order]], axis=1)

    # Convert to CSV for SageMaker (label column must be first)
    # train_combined = pd.concat([y_train, X_train], axis=1)
    # val_combined = pd.concat([y_val, X_val], axis=1)
    # print("Train Columns:", train_combined.columns)
    # print("Validation Columns:", val_combined.columns)
    #
    # # Explicitly set column order
    # train_combined = train_combined[['on_time'] + list(X_train.columns)]
    # val_combined = val_combined[['on_time'] + list(X_train.columns)]



    # Ensure 'on_time' is the first column
    train_combined = pd.concat([y_train, X_train[feature_order]], axis=1)
    val_combined = pd.concat([y_val, X_val[feature_order]], axis=1)

    # Ensure proper order before saving
    train_combined = train_combined[['on_time'] + feature_order]
    val_combined = val_combined[['on_time'] + feature_order]

    train_csv_path = "train.csv"
    val_csv_path = "validation.csv"

    train_combined.reset_index(drop=True, inplace=True)
    val_combined.reset_index(drop=True, inplace=True)

    print(train_combined.head())
    print(val_combined.head())

    train_combined.to_csv(train_csv_path, index=False, header=False)
    val_combined.to_csv(val_csv_path, index=False, header=False)

    with open(train_csv_path, "r") as f:
        for _ in range(5):
            print(f.readline())

    print("Train CSV Preview:\n", train_combined.head())
    print("Validation CSV Preview:\n", val_combined.head())

    if isinstance(train_combined.index, pd.MultiIndex):
        train_combined.reset_index(inplace=True)
    if isinstance(val_combined.index, pd.MultiIndex):
        val_combined.reset_index(inplace=True)

    train_combined.to_csv(train_csv_path, index=False, header=False)
    val_combined.to_csv(val_csv_path, index=False, header=False)

    print("Train Index:", train_combined.index)
    print("Validation Index:", val_combined.index)


    print("X_val Columns:", X_val.columns)
    print("Feature Order:", feature_order)

    print("Train Data Preview:\n", train_csv_path)
    print("Validation Data Preview:\n", train_csv_path)

    train_s3_path = sagemaker_session.upload_data(path=train_csv_path, bucket=bucket, key_prefix=prefix + "/train")
    val_s3_path = sagemaker_session.upload_data(path=val_csv_path, bucket=bucket, key_prefix=prefix + "/validation")

    return train_s3_path, val_s3_path, val_csv_path, X_val, y_val



# Function to Train XGBoost Model
def train_xgboost(train_s3_path, val_s3_path):
    # train_combined = pd.concat([y_train, X_train], axis=1)
    # val_combined = pd.concat([y_val, X_val], axis=1)
    # train_csv_path = "train.csv"
    # val_csv_path = "validation.csv"
    # train_combined.to_csv(train_csv_path, index=False, header=False)
    # val_combined.to_csv(val_csv_path, index=False, header=False)
    #
    # train_s3_path = sagemaker_session.upload_data(path=train_csv_path, bucket=bucket, key_prefix=prefix + "/train")
    # val_s3_path = sagemaker_session.upload_data(path=val_csv_path, bucket=bucket, key_prefix=prefix + "/validation")

    xgboost_image_uri = retrieve("xgboost", region, "1.5-1")
    xgb = sagemaker.estimator.Estimator(
        image_uri=xgboost_image_uri,
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        output_path=f"s3://{bucket}/{prefix}/output",
        sagemaker_session=sagemaker_session
    )

    xgb.set_hyperparameters(
        objective="binary:logistic",
        num_round=100,
        max_depth=5,
        eta=0.2,
        subsample=0.8,
        eval_metric="auc"
    )

    train_input = TrainingInput(train_s3_path, content_type="text/csv")
    val_input = TrainingInput(val_s3_path, content_type="text/csv")
    xgb.fit({"train": train_input, "validation": val_input})
    return xgb


# Function to Evaluate Model
def evaluate_model(xgb_obj, val_csv_path, s3_bucket, X_val, y_true):
    import xgboost as xgb
    is_model_worse = False
    local_model_path = "xgboost_model.tar.gz"
    s3_client = boto3.client('s3')

    # Extract bucket name and key from S3 path
    bucket_name = xgb_obj.model_data.split('/')[2]
    key = "/".join(xgb_obj.model_data.split('/')[3:])

    print(f"Downloading model from {xgb_obj.model_data} to {local_model_path}...")
    s3_client.download_file(bucket_name, key, local_model_path)

    # Unzip model
    os.system(f"tar -xvzf {local_model_path}")

    # Load the XGBoost model (assuming model is named 'xgboost-model')
    model = xgb.Booster()
    model.load_model("xgboost-model")
    val_data = pd.read_csv(val_csv_path, header=None)
    y_true = val_data.iloc[:, 0]
    print(f"YTRUE + ${y_true}")
    # model = xgb.Booster()
    # model.load_model(xgb_obj.model_data)
    print(f"XVAL + ${X_val.head()}")
    # Convert validation data to DMatrix
    dmatrix_val = xgb.DMatrix(X_val)

    # Generate predictions
    y_pred_prob = model.predict(dmatrix_val)


    # Convert probabilities to binary classification (threshold 0.5)
    y_pred = (y_pred_prob).astype(int)
    #y_pred = val_data.iloc[:, 1]
    print(y_pred)
    print(y_true.head())
    mse_new = mean_squared_error(y_true, y_pred)

    # Calculate Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"New Model Accuracy: {accuracy:.4f}")

    # Display Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print(f"New Model MSE: {mse_new}")

    metrics_file = "model_metrics.json"
    try:
        s3_client = boto3.client('s3')
        s3_client.download_file(s3_bucket, f"baseline_models/{metrics_file}", metrics_file)
        with open(metrics_file, "r") as f:
            baseline_metrics = json.load(f)
        mse_baseline = baseline_metrics.get("mse", float("inf"))
        print(f"Baseline MSE: {mse_baseline}")
        if mse_new > mse_baseline:
            is_model_worse = True
            print("New model performs worse than baseline. Exiting pipeline.")
    except Exception as e:
        print(f"Baseline metrics not found or error occurred: {e}")

    with open(metrics_file, "w") as f:
        json.dump({"mse": mse_new}, f, indent=4)
    s3_client.upload_file(metrics_file, s3_bucket, f"xgboost_models/{metrics_file}")

    return is_model_worse


# Function to Deploy Model
def deploy_model(xgb):
    endpoint_name = "flight-delay-xgboost-endpoint"
    predictor = xgb.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.xlarge",
        endpoint_name=endpoint_name
    )
    return predictor, endpoint_name

# Function to Store Model Artifacts
def store_model(xgb, s3_bucket):
    model_artifact_path = xgb.model_data  # S3 path of trained model
    s3_client = boto3.client('s3')

    local_model_path = "xgboost_model.tar.gz"

    # Extract bucket name and key from S3 path
    bucket_name = xgb.model_data.split('/')[2]
    key = "/".join(xgb.model_data.split('/')[3:])
    print(f"Downloading model from {xgb.model_data} to {local_model_path}...")
    s3_client.download_file(bucket_name, key, local_model_path)

    s3_model_path = f"xgboost_models/xgboost_model.tar.gz"
    s3_client.upload_file(local_model_path, s3_bucket, s3_model_path)
    print(f"✅ Model stored at s3://{s3_bucket}/{s3_model_path}")
    return f"s3://{s3_bucket}/{s3_model_path}"


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
def main():
    is_model_worse = False
    config = load_config()
    bucket = config['bucket']
    s3_staging_dir = config["s3_staging_dir"]
    dev_feature_store_table = get_feature_store_table_name(config["dev_fg_name"])
    prod_feature_store_table = get_feature_store_table_name(config["prod_fg_name"])
    df = get_feature_store_data(dev_feature_store_table, s3_staging_dir, region)
    train_s3_path, val_s3_path, val_csv_path, X_val, y_val = prepare_data(df)
    xgb = train_xgboost(train_s3_path, val_s3_path)

    is_model_worse = evaluate_model(xgb, val_csv_path, bucket, X_val, y_val)
    model_s3_path = store_model(xgb, bucket)
    if is_model_worse:
        raise RuntimeError("Model performing worse, exiting pipeline!")

    print(f"✅ Model training complete. Stored at: {model_s3_path}")


if __name__ == "__main__":
    main()
