import boto3

# AWS Configurations
region = "us-east-1"
role_arn = "arn:aws:iam::701586278703:role/service-role/AmazonSageMaker-ExecutionRole-20250214T203663"
repo_owner = "jshah88"
repo_name = "https://github.com/jshah88/AAI540-Group5"
repo_branch = "users/jshah/ci-cd"

# Define CodePipeline Name
pipeline_name = "XGBoostModelPipeline"

# Define Stages
stages = [
    {"name": "SetupEnv", "project": "XGBoostSetupEnv"},
    {"name": "FeatureStore", "project": "XGBoostFeatureStore"},
    {"name": "Preprocessing", "project": "XGBoostPreprocessing"},
    {"name": "BaselineModel", "project": "XGBoostBaselineModel"},
    {"name": "Training", "project": "XGBoostTraining"},
    {"name": "Deployment", "project": "XGBoostDeployment"},
    {"name": "Monitoring", "project": "XGBoostMonitoring"},
]

# Initialize Boto3 Clients
codepipeline = boto3.client("codepipeline", region_name=region)

# Create CodePipeline
response = codepipeline.create_pipeline(
    pipeline={
        "name": pipeline_name,
        "roleArn": role_arn,
        "artifactStore": {"type": "S3", "location": "sagemaker-us-east-1-701586278703"},
        "stages": [
            {
                "name": "Source",
                "actions": [
                    {
                        "name": "SourceAction",
                        "actionTypeId": {
                            "category": "Source",
                            "owner": "ThirdParty",
                            "provider": "GitHub",
                            "version": "1",
                        },
                        "configuration": {
                            "Owner": repo_owner,
                            "Repo": repo_name,
                            "Branch": repo_branch,
                            "OAuthToken": "codebuild-managed-token"
                        },
                        "outputArtifacts": [{"name": "SourceOutput"}],
                    }
                ],
            }
        ]
        + [
            {
                "name": stage["name"],
                "actions": [
                    {
                        "name": f"{stage['name']}Action",
                        "actionTypeId": {
                            "category": "Build",
                            "owner": "AWS",
                            "provider": "CodeBuild",
                            "version": "1",
                        },
                        "configuration": {
                            "ProjectName": stage["project"],
                        },
                        "inputArtifacts": [{"name": "SourceOutput"}],
                        "outputArtifacts": [{"name": f"{stage['name']}Output"}],
                    }
                ],
            }
            for stage in stages
        ],
    }
)

print(f"🚀 CodePipeline '{pipeline_name}' created successfully!")
