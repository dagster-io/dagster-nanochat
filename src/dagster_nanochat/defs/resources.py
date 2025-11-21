import dagster as dg
from dagster_aws.s3 import S3Resource

from dagster_nanochat.defs.runpod_resource import RunPodResource
from dagster_nanochat.defs.serverless_resource import ServerlessResource

s3_resource = S3Resource(
    region_name="us-east-1",
    aws_access_key_id=dg.EnvVar("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=dg.EnvVar("AWS_SECRET_ACCESS_KEY"),
)


runpod_serverless_resource = ServerlessResource(
    api_key=dg.EnvVar("RUNPOD_API_KEY"),
    timeout=120,
)

runpod_training_resource = RunPodResource(
    api_key=dg.EnvVar("RUNPOD_API_KEY"),
    gpu_count=2,
    gpu_type_id="NVIDIA A40",
    env_variables={
        "RUNPOD_SECRET_AWS_ACCESS_KEY_ID": dg.EnvVar("AWS_ACCESS_KEY_ID"),
        "RUNPOD_SECRET_AWS_SECRET_ACCESS_KEY": dg.EnvVar("AWS_SECRET_ACCESS_KEY"),
        "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
        "TORCHELASTIC_ERROR_FILE": "/workspace/error.json",
    },
)


@dg.definitions
def resources():
    return dg.Definitions(
        resources={
            "runpod": runpod_training_resource,
            "serverless": runpod_serverless_resource,
            "s3": s3_resource,
        }
    )
