# baseline_model.py
import boto3
import pandas as pd
import numpy as np
import joblib
import os
import time
import sagemaker
from pyathena import connect
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from utils import setup_logger, load_config
import json

logger = setup_logger("baseline_model")


# Initialize Athena connection
def setup_athena_connection(s3_staging_dir, region):
    return connect(s3_staging_dir=s3_staging_dir, region_name=region)


# Query data from Athena Feature Store
def get_feature_store_data(conn, table_name):
    print(f"Fetching data from Feature Store Table: {table_name}")
    query = f'SELECT * FROM "sagemaker_featurestore"."{table_name}";'
    return pd.read_sql(query, conn)


# Preprocessing Functions
def preprocess_data(df, features, target):
    df = df.dropna()
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Model Evaluation and Metrics Storage
def evaluate_and_store_metrics(model_name, y_test, y_pred, s3_bucket, metrics_file="model_metrics.json"):
    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }

    print(f"{model_name} Metrics:", metrics)

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_file}")

    # Upload to S3
    s3_client = boto3.client('s3')
    s3_client.upload_file(metrics_file, s3_bucket, f"models/{metrics_file}")
    print(f"Metrics uploaded to s3://{s3_bucket}/models/{metrics_file}")


# Save model and upload to S3
def save_and_upload_model(model, filename, s3_bucket):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

    s3_client = boto3.client('s3')
    s3_client.upload_file(filename, s3_bucket, f"models/{filename}")
    print(f"Model uploaded to s3://{s3_bucket}/models/{filename}")


# Linear Regression Model
def train_linear_regression(X_train, X_test, y_train, y_test, s3_bucket):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    evaluate_and_store_metrics("LinearRegression", y_test, y_test_pred, s3_bucket)
    save_and_upload_model(model, "linear_regression_model.pkl", s3_bucket)
    return model
#
# def train_linear_regression(X_train, y_train):
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     logger.info("Linear Regression model trained.")
#     return model
#
# def evaluate_regression(model, X_test, y_test):
#     predictions = model.predict(X_test)
#     mse = mean_squared_error(y_test, predictions)
#     r2 = r2_score(y_test, predictions)
#     logger.info(f"Regression Evaluation - MSE: {mse:.4f}, R2: {r2:.4f}")
#     return mse, r2
#
# def train_logistic_regression(X_train, y_train):
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     model = LogisticRegression()
#     model.fit(X_train_scaled, y_train)
#     logger.info("Logistic Regression model trained.")
#     return model, scaler
#
# def evaluate_classification(model, scaler, X_test, y_test):
#     X_test_scaled = scaler.transform(X_test)
#     prob_pred = model.predict_proba(X_test_scaled)[:, 1]
#     y_pred = (prob_pred >= 0.5).astype(int)
#     accuracy = accuracy_score(y_test, y_pred)
#     logger.info(f"Classification Accuracy: {accuracy:.4f}")
#     logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))
#     logger.info("Classification Report:\n" + classification_report(y_test, y_pred))
#     return accuracy
#
# def visualize_regression(y_true, y_pred, title="Regression Predictions"):
#     plt.scatter(y_true, y_pred, alpha=0.5)
#     plt.xlabel("Actual")
#     plt.ylabel("Predicted")
#     plt.title(title)
#     plt.grid()
#     plt.show()

def save_model(model, file_path):
    joblib.dump(model, file_path)
    logger.info(f"Model saved at {file_path}")


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


if __name__ == "__main__":
    config = load_config()
    region = config['region']
    bucket = config['bucket']
    database_name = config["athena_database"]
    raw_table_name = config["raw_table_name"]
    s3_staging_dir = config["s3_staging_dir"]
    dev_table_name = config["dev_table_name"]
    prod_table_name = config["prod_table_name"]
    role = config["sagemaker_execution_role"]
    dev_fg_table_name = get_feature_store_table_name(config["dev_fg_name"])
    conn = setup_athena_connection(s3_staging_dir, region)
    df = get_feature_store_data(conn, dev_fg_table_name)

    features = [
        "arr_flights", "arr_del15", "carrier_ct", "weather_ct", "nas_ct",
        "security_ct", "late_aircraft_ct", "arr_cancelled", "arr_diverted",
        "arr_delay", "carrier_delay", "weather_delay", "nas_delay",
        "security_delay", "late_aircraft_delay", "delay_rate"
    ]
    target = "on_time"


    X_train, X_test, y_train, y_test = preprocess_data(df, features, target)

    train_linear_regression(X_train, X_test, y_train, y_test, bucket)

