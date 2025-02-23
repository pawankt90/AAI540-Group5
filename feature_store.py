# feature_store.py
import time
import boto3
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup
from botocore.exceptions import ClientError
from utils import setup_logger, load_config
from pyathena import connect
import pandas as pd

logger = setup_logger("feature_store")

def setup_athena_connection(s3_staging_dir, region):
    return connect(s3_staging_dir=s3_staging_dir, region_name=region)

# Feature Store Management
def delete_feature_group(sagemaker_client, feature_group_name):
    try:
        existing_groups = sagemaker_client.list_feature_groups()['FeatureGroupSummaries']
        existing_group_names = [fg['FeatureGroupName'] for fg in existing_groups]
        if feature_group_name in existing_group_names:
            sagemaker_client.delete_feature_group(FeatureGroupName=feature_group_name)
            while True:
                existing_groups = sagemaker_client.list_feature_groups()['FeatureGroupSummaries']
                existing_group_names = [fg['FeatureGroupName'] for fg in existing_groups]
                if feature_group_name not in existing_group_names:
                    break
                time.sleep(5)
    except Exception as e:
        print(f"Error deleting Feature Group '{feature_group_name}': {e}")

# def create_feature_group(sagemaker_client, feature_group_name, data_df, s3_uri, role):
#     """Create a new feature group with a schema inferred from data_df."""
#     existing = list_feature_groups(sagemaker_client)
#     if feature_group_name in existing:
#         logger.info(f"Feature Group '{feature_group_name}' already exists.")
#         return
#     feature_definitions = [{"FeatureName": "event_time", "FeatureType": "Fractional"}] + [
#         {"FeatureName": col, "FeatureType": "String" if data_df[col].dtype == "object" else "Integral"}
#         for col in data_df.columns if col != "event_time"
#     ]
#     try:
#         sagemaker_client.create_feature_group(
#             FeatureGroupName=feature_group_name,
#             RecordIdentifierFeatureName="record_id",
#             EventTimeFeatureName="event_time",
#             FeatureDefinitions=feature_definitions,
#             OnlineStoreConfig={"EnableOnlineStore": True},
#             OfflineStoreConfig={
#                 "S3StorageConfig": {"S3Uri": s3_uri},
#                 "DisableGlueTableCreation": False,
#             },
#             RoleArn=role
#         )
#         logger.info(f"Feature Group '{feature_group_name}' creation initiated.")
#     except Exception as e:
#         logger.error(f"Error creating feature group '{feature_group_name}': {e}")
#         raise e
#     # Wait until the feature group is active
#     while True:
#         try:
#             status = sagemaker_client.describe_feature_group(FeatureGroupName=feature_group_name)["FeatureGroupStatus"]
#             logger.info(f"Feature Group '{feature_group_name}' status: {status}")
#             if status == "Created":
#                 break
#         except Exception as e:
#             logger.error(f"Error checking status: {e}")
#         time.sleep(5)
#     logger.info(f"Feature Group '{feature_group_name}' is active.")


def create_feature_group(sagemaker_client, feature_group_name, df, s3_uri, role):
    existing_groups = sagemaker_client.list_feature_groups()['FeatureGroupSummaries']
    existing_group_names = [fg['FeatureGroupName'] for fg in existing_groups]
    if feature_group_name not in existing_group_names:
        feature_group_definition = {
            "FeatureGroupName": feature_group_name,
            "RecordIdentifierFeatureName": "record_id",
            "EventTimeFeatureName": "event_time",
            "FeatureDefinitions": [
                {"FeatureName": "event_time", "FeatureType": "Fractional"}
            ] + [
                {"FeatureName": col, "FeatureType": "String" if df[col].dtype == "string" else "Integral"}
                for col in df.columns if col != "event_time"
            ],
            "OnlineStoreConfig": {"EnableOnlineStore": True},
            "OfflineStoreConfig": {"S3StorageConfig": {"S3Uri": s3_uri}},
            "RoleArn": role,
        }
        sagemaker_client.create_feature_group(**feature_group_definition)
        while True:
            status = sagemaker_client.describe_feature_group(FeatureGroupName=feature_group_name)["FeatureGroupStatus"]
            if status == "Created":
                break
            time.sleep(5)
        time.sleep(60)
# def ingest_record(featurestore_runtime, feature_group_name, record_dict):
#     """Insert a single record into the specified feature group."""
#     record = {
#         "FeatureGroupName": feature_group_name,
#         "Record": [{"FeatureName": k, "ValueAsString": str(v)} for k, v in record_dict.items()]
#     }
#     try:
#         featurestore_runtime.put_record(**record)
#         logger.info(f"Inserted record into feature group '{feature_group_name}'.")
#     except Exception as e:
#         logger.error(f"Error inserting record: {e}")


def insert_single_record(featurestore_runtime, feature_group_name, df):
    df["event_time"] = time.time()
    single_record = df.iloc[0].to_dict()
    record = {
        "FeatureGroupName": feature_group_name,
        "Record": [{"FeatureName": key, "ValueAsString": str(value)} for key, value in single_record.items()]
    }
    featurestore_runtime.put_record(**record)

def bulk_ingest(feature_group, df):
    feature_group.ingest(data_frame=df[1:], max_workers=5, wait=True)

# Query data from Athena
def query_athena_data(conn, database_name, table_name):
    query = f"SELECT * FROM {database_name}.{table_name}"
    return pd.read_sql(query, conn)

# Data Preprocessing Functions
def cast_object_to_string(df):
    for col in df.columns:
        if df.dtypes[col] == "object":
            df[col] = df[col].astype("string")

def convert_numeric_to_int(df):
    for col in df.select_dtypes(include=["int", "float"]).columns:
        if col != "event_time":
            df[col] = df[col].astype("int64")

def is_on_time(row):
    return 1 if row['arr_del15'] == 0 and row['arr_cancelled'] == 0 else 0

def preprocess_data(df):
    cast_object_to_string(df)
    convert_numeric_to_int(df)
    df['on_time'] = df.apply(is_on_time, axis=1)
    df['event_time'] = time.time()
    df['record_id'] = df.index.astype("string")
    return df


if __name__ == "__main__":
    # Example usage (replace with actual data loading)

    config = load_config()
    region = config['region']
    bucket = config['bucket']
    database_name = config["athena_database"]
    raw_table_name = config["raw_table_name"]
    s3_staging_dir = config["s3_staging_dir"]
    dev_table_name = config["dev_table_name"]
    prod_table_name = config["prod_table_name"]
    role = config["sagemaker_execution_role"]

    from utils import init_aws_session
    session, sagemaker_session = init_aws_session()
    sagemaker_client = boto3.client("sagemaker", region_name=region)

    conn = setup_athena_connection(s3_staging_dir, region)

    dev_df = query_athena_data(conn, database_name, dev_table_name)
    prod_df = query_athena_data(conn, database_name, prod_table_name)

    dev_df = preprocess_data(dev_df)
    prod_df = preprocess_data(prod_df)

    featurestore_runtime = boto3.client("sagemaker-featurestore-runtime", region_name=region)

    dev_fg_name = "airline_delay_features_dev"
    prod_fg_name = "airline_delay_features_prod"
    dev_s3_uri = f"s3://{bucket}/feature-store/dev/"
    prod_s3_uri = f"s3://{bucket}/feature-store/prod/"

    delete_feature_group(sagemaker_client, dev_fg_name)
    delete_feature_group(sagemaker_client, prod_fg_name)

    create_feature_group(sagemaker_client, dev_fg_name, dev_df, dev_s3_uri, role)
    create_feature_group(sagemaker_client, prod_fg_name, prod_df, prod_s3_uri, role)

    insert_single_record(featurestore_runtime, dev_fg_name, dev_df)
    insert_single_record(featurestore_runtime, prod_fg_name, prod_df)

    dev_feature_group = sagemaker.feature_store.feature_group.FeatureGroup(name=dev_fg_name,
                                                                           sagemaker_session=sagemaker_session)
    prod_feature_group = sagemaker.feature_store.feature_group.FeatureGroup(name=prod_fg_name,
                                                                            sagemaker_session=sagemaker_session)

    bulk_ingest(dev_feature_group, dev_df)
    bulk_ingest(prod_feature_group, prod_df)

    # df should be loaded here
    # role = "arn:aws:iam::123456789012:role/YourSageMakerRole"
    # s3_uri = f"s3://{sagemaker_session.default_bucket()}/feature-store/dev/"
    # create_feature_group(sagemaker_client, "your_feature_group_dev", df, s3_uri, role)
    pass
