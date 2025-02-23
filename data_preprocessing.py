# data_preprocessing.py
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import setup_logger, load_config
import boto3

logger = setup_logger("data_preprocessing")

def split_data(df, target_col, test_size=0.2, random_state=42):
    """Split DataFrame into train and test sets."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logger.info(f"Data split: {len(X_train)} train and {len(X_test)} test samples.")
    return X_train, X_test, y_train, y_test

def export_to_csv(df, file_path, header=False, index=False):
    """Export DataFrame to CSV."""
    df.to_csv(file_path, header=header, index=index)
    logger.info(f"Data exported to {file_path}.")

def plot_yearly_trends(df, group_col="year", cols=["arr_flights", "arr_del15", "arr_cancelled"]):
    """Plot trends over years for specified columns."""
    summary = df.groupby(group_col)[cols].sum().reset_index()
    plt.figure(figsize=(14, 7))
    for col in cols:
        plt.plot(summary[group_col], summary[col], label=col)
    plt.title("Yearly Trends")
    plt.xlabel("Year")
    plt.ylabel("Counts")
    plt.legend()
    plt.grid()
    plt.show()


def get_raw_data(database_name, raw_table_name, s3_staging_dir, region):
    from pyathena import connect
    # Set up Athena connection with stored values
    conn = connect(s3_staging_dir=s3_staging_dir, region_name=region)

    # Query Athena for the filtered dataset
    query = f"SELECT * FROM {database_name}.{raw_table_name}"

    print("Querying Athena...")

    try:
        airline_data = pd.read_sql(query, conn)
        print("Data loaded successfully!")
        print(f"Number of rows: {len(airline_data)}")
        return airline_data
    except Exception as e:
        print(f"Failed to execute Athena query: {e}")

def analyze_yearly_trends(df):
    yearly_summary = df.groupby('year')[['arr_flights', 'arr_del15', 'arr_cancelled']].sum().reset_index()
    yearly_summary['overall_delay_rate'] = (yearly_summary['arr_del15'] / yearly_summary['arr_flights']).fillna(0) * 100
    return yearly_summary

def analyze_carrier_performance(df):
    carrier_performance = df.groupby('carrier')[['arr_flights', 'arr_del15']].sum().reset_index()
    carrier_performance['overall_delay_rate'] = (carrier_performance['arr_del15'] / carrier_performance['arr_flights']).fillna(0) * 100
    return carrier_performance.sort_values('overall_delay_rate', ascending=False).head(10)

def analyze_airport_performance(df):
    airport_performance = df.groupby('airport')[['arr_flights', 'arr_del15', 'arr_cancelled', 'arr_diverted']].sum().reset_index()
    airport_performance['overall_delay_rate'] = (airport_performance['arr_del15'] / airport_performance['arr_flights']).fillna(0) * 100
    return airport_performance.sort_values('overall_delay_rate', ascending=False).head(10)

def analyze_delay_reasons(df):
    delay_reasons = df.groupby('year')[['carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay']].sum().reset_index()
    return delay_reasons

# Feature Engineering Functions
def is_on_time(row):
    return 0 if row['arr_del15'] > 0 or row['arr_cancelled'] > 0 else 1

def clean_data(df):
    df = df.fillna(0)
    df['delay_rate'] = (df['arr_del15'] / df['arr_flights']).fillna(0) * 100
    df['on_time'] = df.apply(is_on_time, axis=1)
    logger.info("Data cleaned.")
    return df

def filter_top_carriers(df):
    carrier_flights = df.groupby('carrier')['arr_flights'].sum().reset_index()
    top_10_carriers = carrier_flights.nlargest(10, 'arr_flights')['carrier'].tolist()
    return df[df['carrier'].isin(top_10_carriers)]

def split_data_temporally(df):
    df = df.sort_values(by=['year', 'month'])
    dev_data = df[(df['year'] >= 2004) & (df['year'] <= 2015)]
    prod_data = df[(df['year'] >= 2016) & (df['year'] <= 2024)]
    return dev_data, prod_data


def save_and_upload_to_s3(df, bucket, prefix):
    file_path = f"{prefix}.csv"
    df.to_csv(file_path, index=False)
    s3 = boto3.client('s3')
    s3.upload_file(file_path, bucket, f"data/{prefix}/{prefix}.csv")
    print(f"Uploaded {file_path} to s3://{bucket}/data/{prefix}/{prefix}.csv")

# Athena Table Management
def drop_existing_athena_tables(conn, database_name):
    tables = ['development_data', 'production_data']
    for table in tables:
        query = f"DROP TABLE IF EXISTS {database_name}.{table};"
        with conn.cursor() as cursor:
            cursor.execute(query)
        print(f"Dropped table {table} if it existed.")
def setup_athena_tables(conn, database_name, bucket):
    from pyathena import connect
    conn = connect(s3_staging_dir=s3_staging_dir, region_name=region)
    tables = {'development_data': 'development_data', 'production_data': 'production_data'}
    create_table_query_template = """
    CREATE EXTERNAL TABLE IF NOT EXISTS {database}.{table_name} (
        year INT,
        month INT,
        carrier STRING,
        airport STRING,
        arr_flights INT,
        arr_del15 INT,
        carrier_ct INT,
        weather_ct INT,
        nas_ct INT,
        security_ct INT,
        late_aircraft_ct INT,
        arr_cancelled INT,
        arr_diverted INT,
        arr_delay INT,
        carrier_delay INT,
        weather_delay INT,
        nas_delay INT,
        security_delay INT,
        late_aircraft_delay INT,
        delay_rate FLOAT,
        on_time INT
    )
    ROW FORMAT DELIMITED
    FIELDS TERMINATED BY ','
    STORED AS TEXTFILE
    LOCATION 's3://{bucket}/data/{folder_name}/'
    TBLPROPERTIES ('skip.header.line.count'='1');
    """
    for table, folder in tables.items():
        query = create_table_query_template.format(database=database_name, table_name=table, bucket=bucket, folder_name=folder)
        pd.read_sql(query, conn)
        print(f"Athena table {table} created successfully!")


def count_rows_in_tables(conn, database_name, dev_table_name, prod_table_name):
    count_dev_query = f"SELECT COUNT(*) FROM {database_name}.{dev_table_name};"
    count_prod_query = f"SELECT COUNT(*) FROM {database_name}.{prod_table_name};"
    dev_count_df = pd.read_sql(count_dev_query, conn)
    prod_count_df = pd.read_sql(count_prod_query, conn)
    print(f"Development Data Row Count: {dev_count_df.iloc[0, 0]}")
    print(f"Production Data Row Count: {prod_count_df.iloc[0, 0]}")

def setup_athena_connection(s3_staging_dir, region):
    from pyathena import connect
    return connect(s3_staging_dir=s3_staging_dir, region_name=region)

if __name__ == "__main__":
    config = load_config()
    region = config['region']
    bucket = config['bucket']
    database_name = config["athena_database"]
    raw_table_name = config["raw_table_name"]
    s3_staging_dir = config["s3_staging_dir"]
    conn = setup_athena_connection(s3_staging_dir, region)
    drop_existing_athena_tables(conn, database_name)

    airline_data = get_raw_data(database_name,raw_table_name,s3_staging_dir,region)

    df = clean_data(airline_data)
    df = filter_top_carriers(df)

    print("Data Retrieved Successfully")
    print(df.head())

    yearly_trends = analyze_yearly_trends(df)
    carrier_performance = analyze_carrier_performance(df)
    airport_performance = analyze_airport_performance(df)
    delay_reasons = analyze_delay_reasons(df)

    print("Yearly Trends:\n", yearly_trends.head())
    print("Carrier Performance:\n", carrier_performance)
    print("Airport Performance:\n", airport_performance)
    print("Delay Reasons:\n", delay_reasons)

    dev_data, prod_data = split_data_temporally(df)
    save_and_upload_to_s3(dev_data, bucket, "development_data")
    save_and_upload_to_s3(prod_data, bucket, "production_data")

    setup_athena_tables(conn, database_name, bucket)
    count_rows_in_tables(conn,database_name,"development_data","production_data")

    #X_train, X_test, y_train, y_test = split_data(cleaned_df, target_col="on_time")
    #export_to_csv(cleaned_df, "cleaned_data.csv")
    #plot_yearly_trends(cleaned_df)
