import os

import kfp
import kfp.components as comp
import kfp.dsl as dsl


def say_hello(user_name: str) -> None:
    """Function to say hello."""
    print(f"Hello {user_name}, Welcome!")


say_hello_op = kfp.components.create_component_from_func(
    say_hello, base_image="python:3.10.9-slim-buster"
)


def download_file_from_gcs_bucket(
    gcs_bucket_name: str, file_name: str, downloaded_path: comp.OutputPath()
):
    """Function to download data from gcs bucket."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(gcs_bucket_name)
    blob = bucket.blob(file_name)
    blob.download_to_filename(downloaded_path)


download_file_from_gcs_bucket_op = kfp.components.create_component_from_func(
    download_file_from_gcs_bucket,
    output_component_file="download_file_from_gcs_bucket.yaml",
    base_image="eu.gcr.io/kubeflow-bg-experiment/hello_world:latest",
    # # alternatively,
    # base_image="python:3.10.9-buster",
    # packages_to_install=["google-cloud-storage~=2.7.0"],
)


@dsl.pipeline(
    name="Hello world pipeline",
    description="Hello world pipeline",
)
def hello_world_pipeline(user_name, gcs_bucket_name, file_name):
    """Function to run hello world pipeline."""
    say_hello_task = say_hello_op(user_name)
    say_hello_task.container.set_image_pull_policy("Always")
    say_hello_task.set_caching_options(False)
    say_hello_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    download_file_from_gcs_bucket_task = download_file_from_gcs_bucket_op(
        gcs_bucket_name=gcs_bucket_name,
        file_name=file_name,
    )
    download_file_from_gcs_bucket_task.container.set_image_pull_policy("Always")
    download_file_from_gcs_bucket_task.set_caching_options(False)
    download_file_from_gcs_bucket_task.execution_options.caching_strategy.max_cache_staleness = (
        "P0D"
    )
    # # try pipeline task on nodes based on arm64, possible in future after this issue fix: https://github.com/kubeflow/kubeflow/issues/2337
    # download_file_from_gcs_bucket_task.add_toleration(
    #     {"key": "kubernetes.io/arch", "value": "arm64"}
    # )
    # download_file_from_gcs_bucket_task.add_node_selector_constraint(
    #     "kubernetes.io/arch", "arm64"
    # )


kfp.compiler.Compiler().compile(
    pipeline_func=hello_world_pipeline, package_path="hello_world_pipeline.yaml"
)

client = kfp.Client(host="http://localhost:3000")
client.create_run_from_pipeline_func(
    hello_world_pipeline,
    arguments={
        "user_name": f"{os.environ.get('USER_NAME')}",
        "gcs_bucket_name": f"{os.environ.get('GCS_STORAGE_BUCKET_NAME')}",
        "file_name": f"{os.environ.get('FILE_NAME')}",
    },
)
