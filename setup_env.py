# setup_env.py
import boto3
import sagemaker
from botocore.exceptions import ClientError
from utils import init_aws_session, setup_logger, load_config,copy_to_s3

logger = setup_logger("setup_env")

def verify_s3_bucket(sagemaker_session):
    """Verify that the default S3 bucket exists and return its name."""
    bucket = sagemaker_session.default_bucket()
    region = sagemaker_session.boto_session.region_name
    s3 = boto3.client("s3", region_name=region)
    try:
        s3.head_bucket(Bucket=bucket)
        logger.info(f"Verified S3 bucket: {bucket}")
    except ClientError as e:
        logger.error(f"Error: Cannot access bucket {bucket}: {e}")
        raise e
    return bucket

def create_athena_database(database_name, bucket, region):
    """Creates an Athena database if it does not exist."""
    from pyathena import connect
    s3_staging_dir = f"s3://{bucket}/athena/staging"
    conn = connect(region_name=region, s3_staging_dir=s3_staging_dir)
    cursor = conn.cursor()
    statement = f"CREATE DATABASE IF NOT EXISTS {database_name}"
    cursor.execute(statement)
    logger.info(f"Athena database '{database_name}' is ready.")
    return s3_staging_dir

def create_athena_dataset_table(database_name, s3_staging_dir, region, s3_csv_private_path):

    # Set Athena parameters
    from pyathena import connect
    import pandas as pd
    raw_table_name = "airline_delay_cause_csv_raw"

    conn = connect(region_name=region, s3_staging_dir=s3_staging_dir)
    # SQL statement to execute
    statement = """CREATE EXTERNAL TABLE IF NOT EXISTS {}.{}(
             year int,
             month int,
             carrier string,
             airport string,
             arr_flights int,
             arr_del15 int,
             carrier_ct float,
             weather_ct float,
             nas_ct float,
             security_ct float,
             late_aircraft_ct float,
             arr_cancelled int,
             arr_diverted int,
             arr_delay int,
             carrier_delay int,
             weather_delay int,
             nas_delay int,
             security_delay int,
             late_aircraft_delay int
    ) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\\n' LOCATION '{}'
    TBLPROPERTIES ('compressionType'='gzip', 'skip.header.line.count'='1')""".format(
        database_name, raw_table_name, s3_csv_private_path
    )

    print(statement)

    pd.read_sql(statement, conn)
    statement = "SHOW TABLES in {}".format(database_name)

    df_show = pd.read_sql(statement, conn)
    print(df_show.head(5))

    carrier = "9E"

    statement = """SELECT * FROM {}.{}
        WHERE carrier = '{}' LIMIT 100""".format(
        database_name, raw_table_name, carrier
    )
    print(statement)
    df = pd.read_sql(statement, conn)
    print(df.head(5))

def main():
    config = load_config()  # Optionally load from config.yaml
    region = config['region']
    session, sagemaker_session = init_aws_session(region)
    bucket = verify_s3_bucket(sagemaker_session)
    database_name = config["athena_database"]
    s3_staging_dir = create_athena_database(database_name, bucket, region)
    #load_data_to_s3
    copy_to_s3("data/Airline_Delay_Cause.csv.gz", config['s3_csv_private_path'] )
    create_athena_dataset_table(database_name,s3_staging_dir,region, s3_csv_private_path=config['s3_csv_private_path'])
    logger.info(f"Environment setup complete. Bucket: {bucket}, Athena staging dir: {s3_staging_dir}")

if __name__ == "__main__":
    main()
