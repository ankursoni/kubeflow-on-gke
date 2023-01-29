import os

import kfp
import kfp.components as comp
import kfp.dsl as dsl


def say_hello(user_name: str) -> None:
    """Function to say hello."""
    print(f"Hello {user_name}, Welcome!")


say_hello_op = kfp.components.create_component_from_func(
    say_hello, base_image="python:3.11.1-slim-buster"
)


@dsl.pipeline(
    name="Hello world pipeline",
    description="Hello world pipeline",
)
def hello_world_pipeline(user_name):
    """Function to run hello world pipeline."""
    say_hello_task = say_hello_op(user_name)
    say_hello_task.set_caching_options(False)
    say_hello_task.execution_options.caching_strategy.max_cache_staleness = "P0D"


kfp.compiler.Compiler().compile(
    pipeline_func=hello_world_pipeline, package_path="hello_world_pipeline.yaml"
)

client = kfp.Client(host="http://localhost:3000")
client.create_run_from_pipeline_func(
    hello_world_pipeline,
    arguments={"user_name": f"{os.environ.get('USER_NAME')}"},
)
