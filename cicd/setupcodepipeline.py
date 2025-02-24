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
        "name": "XGBoostModelPipeline",
        "roleArn": "arn:aws:iam::701586278703:role/service-role/AmazonSageMaker-ExecutionRole-20250214T203663",
        "artifactStore": {
            "type": "S3",
            "location": "sagemaker-us-east-1-701586278703"
        },
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
                            "version": "1"
                        },
                        "runOrder": 1,
                        "configuration": {
                            "Branch": "users/jshah/ci-cd",
                            "OAuthToken": "****",
                            "Owner": "jshah88",
                            "PollForSourceChanges": "false",
                            "Repo": "AAI540-Group5"
                        },
                        "outputArtifacts": [
                            {
                                "name": "SourceOutput"
                            }
                        ],
                        "inputArtifacts": [],
                        "region": "us-east-1"
                    }
                ]
            },
            {
                "name": "SetupEnv",
                "actions": [
                    {
                        "name": "SetupEnvAction",
                        "actionTypeId": {
                            "category": "Build",
                            "owner": "AWS",
                            "provider": "CodeBuild",
                            "version": "1"
                        },
                        "runOrder": 1,
                        "configuration": {
                            "ProjectName": "XGBoostSetupEnv"
                        },
                        "outputArtifacts": [
                            {
                                "name": "SetupEnvOutput"
                            }
                        ],
                        "inputArtifacts": [
                            {
                                "name": "SourceOutput"
                            }
                        ]
                    }
                ]
            },
            {
                "name": "FeatureStore",
                "actions": [
                    {
                        "name": "FeatureStoreAction",
                        "actionTypeId": {
                            "category": "Build",
                            "owner": "AWS",
                            "provider": "CodeBuild",
                            "version": "1"
                        },
                        "runOrder": 1,
                        "configuration": {
                            "ProjectName": "XGBoostFeatureStore"
                        },
                        "outputArtifacts": [
                            {
                                "name": "FeatureStoreOutput"
                            }
                        ],
                        "inputArtifacts": [
                            {
                                "name": "SourceOutput"
                            }
                        ]
                    }
                ]
            },
            {
                "name": "Preprocessing",
                "actions": [
                    {
                        "name": "PreprocessingAction",
                        "actionTypeId": {
                            "category": "Build",
                            "owner": "AWS",
                            "provider": "CodeBuild",
                            "version": "1"
                        },
                        "runOrder": 1,
                        "configuration": {
                            "ProjectName": "XGBoostPreprocessing"
                        },
                        "outputArtifacts": [
                            {
                                "name": "PreprocessingOutput"
                            }
                        ],
                        "inputArtifacts": [
                            {
                                "name": "SourceOutput"
                            }
                        ]
                    }
                ]
            },
            {
                "name": "BaselineModel",
                "actions": [
                    {
                        "name": "BaselineModelAction",
                        "actionTypeId": {
                            "category": "Build",
                            "owner": "AWS",
                            "provider": "CodeBuild",
                            "version": "1"
                        },
                        "runOrder": 1,
                        "configuration": {
                            "ProjectName": "XGBoostBaselineModel"
                        },
                        "outputArtifacts": [
                            {
                                "name": "BaselineModelOutput"
                            }
                        ],
                        "inputArtifacts": [
                            {
                                "name": "SourceOutput"
                            }
                        ]
                    }
                ]
            },
            {
                "name": "Training",
                "actions": [
                    {
                        "name": "TrainingAction",
                        "actionTypeId": {
                            "category": "Build",
                            "owner": "AWS",
                            "provider": "CodeBuild",
                            "version": "1"
                        },
                        "runOrder": 1,
                        "configuration": {
                            "ProjectName": "XGBoostTraining"
                        },
                        "outputArtifacts": [
                            {
                                "name": "TrainingOutput"
                            }
                        ],
                        "inputArtifacts": [
                            {
                                "name": "SourceOutput"
                            }
                        ]
                    }
                ]
            },
            {
                "name": "ManualApproval",
                "actions": [
                    {
                        "name": "ManualApproval",
                        "actionTypeId": {
                            "category": "Approval",
                            "owner": "AWS",
                            "provider": "Manual",
                            "version": "1"
                        },
                        "runOrder": 1,
                        "configuration": {
                            "ExternalEntityLink": "https://studio-d-lzazkcrqxhcb.studio.us-east-1.sagemaker.aws/models/registered-models/XGBoostFlightDelayModelGroup/versions",
                            "IsSummaryRequired": "True"
                        },
                        "outputArtifacts": [],
                        "inputArtifacts": [],
                        "region": "us-east-1"
                    }
                ]
            },
            {
                "name": "Deployment",
                "actions": [
                    {
                        "name": "DeploymentAction",
                        "actionTypeId": {
                            "category": "Build",
                            "owner": "AWS",
                            "provider": "CodeBuild",
                            "version": "1"
                        },
                        "runOrder": 1,
                        "configuration": {
                            "ProjectName": "XGBoostDeployment"
                        },
                        "outputArtifacts": [
                            {
                                "name": "DeploymentOutput"
                            }
                        ],
                        "inputArtifacts": [
                            {
                                "name": "SourceOutput"
                            }
                        ]
                    }
                ]
            }
        ],
        "version": 5,
        "executionMode": "QUEUED",
        "pipelineType": "V2"
    }
)

print(f"🚀 CodePipeline '{pipeline_name}' created successfully!")
