# monitoring.py
import boto3
import sagemaker
from sagemaker.model_monitor import DefaultModelMonitor, DatasetFormat, CronExpressionGenerator
from utils import setup_logger

logger = setup_logger("monitoring")


def create_monitoring_schedule(endpoint_name, output_s3_uri, role, instance_type="ml.m5.xlarge", instance_count=1):
    """Create a monitoring schedule for the specified endpoint."""
    sagemaker_session = sagemaker.Session()
    model_monitor = DefaultModelMonitor(
        role=role,
        instance_type=instance_type,
        instance_count=instance_count,
        max_runtime_in_seconds=3600,
        sagemaker_session=sagemaker_session
    )
    schedule_name = f"{endpoint_name}-monitor"
    cron_expression = CronExpressionGenerator.hourly()
    logger.info(f"Creating monitoring schedule '{schedule_name}' for endpoint '{endpoint_name}'...")
    model_monitor.create_monitoring_schedule(
        monitor_schedule_name=schedule_name,
        endpoint_input=endpoint_name,
        output_s3_uri=output_s3_uri,
        schedule_cron_expression=cron_expression,
        enable_cloudwatch_metrics=True,
    )
    logger.info("Monitoring schedule created.")
    return schedule_name


def run_baseline_job(model_monitor, baseline_dataset, output_s3_uri):
    """Run a baseline suggestion job using the provided dataset."""
    try:
        logger.info("Running baseline job...")
        model_monitor.suggest_baseline(
            baseline_dataset=baseline_dataset,
            dataset_format=DatasetFormat.json(),
            output_s3_uri=output_s3_uri,
            wait=True
        )
        logger.info("Baseline job completed.")
    except Exception as e:
        logger.error(f"Baseline job failed: {e}")
        raise e


if __name__ == "__main__":
    # Example usage
    role = "arn:aws:iam::123456789012:role/YourSageMakerRole"  # Replace accordingly
    session = boto3.session.Session()
    sagemaker_session = sagemaker.Session(session)
    bucket = sagemaker_session.default_bucket()
    endpoint_name = "flight-delay-xgboost-endpoint-single-request"
    output_s3_uri = f"s3://{bucket}/monitoring_results/"

    create_monitoring_schedule(endpoint_name, output_s3_uri, role)
