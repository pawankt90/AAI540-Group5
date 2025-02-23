import boto3

# Load Configuration
region = "us-east-1"
role_arn = "arn:aws:iam::701586278703:role/service-role/AmazonSageMaker-ExecutionRole-20250214T203663"
s3_bucket = "sagemaker-us-east-1-701586278703"
repo_url = "https://github.com/jshah88/AAI540-Group5.git"

# Define Build Projects
projects = [
    {"name": "XGBoostSetupEnv", "script": "buildspec_setup_env.yml"},
    {"name": "XGBoostFeatureStore", "script": "buildspec_feature_store.yml"},
    {"name": "XGBoostPreprocessing", "script": "buildspec_data_preprocessing.yml"},
    {"name": "XGBoostBaselineModel", "script": "buildspec_baseline_model.yml"},
    {"name": "XGBoostTraining", "script": "buildspec_train.yml"},
    {"name": "XGBoostDeployment", "script": "buildspec_deploy.yml"},
    {"name": "XGBoostMonitoring", "script": "buildspec_monitoring.yml"},
]

# Initialize Boto3 Clients
codebuild = boto3.client("codebuild", region_name=region)

def create_build_project(project_name, script_name):
    response = codebuild.update_project(
        name=project_name,
        source={
            "type": "GITHUB",
            "location": repo_url,
            "buildspec": f"{script_name}",  # Assuming buildspec is in repo
        },
        artifacts={"type": "NO_ARTIFACTS"},
        environment={
            "type": "LINUX_CONTAINER",
            "image": "aws/codebuild/standard:5.0",
            "computeType": "BUILD_GENERAL1_SMALL",
            "environmentVariables": [],
        },
        serviceRole=role_arn,
    )
    print(f"✅ Created CodeBuild project: {project_name}")

# Loop through and create all projects
for project in projects:
    create_build_project(project["name"], project["script"])

print("🚀 All CodeBuild projects created successfully!")
