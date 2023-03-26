import os

import kfp
import kfp.components as comp
import kfp.dsl as dsl


def prepare_dataset(
    movies_path: comp.OutputPath(),
    train_path: comp.OutputPath(),
    test_path: comp.OutputPath(),
    unique_movie_titles_path: comp.OutputPath(),
    unique_user_ids_path: comp.OutputPath(),
):
    """Function to prepare dataset."""
    import pickle

    import numpy as np
    import tensorflow as tf
    import tensorflow_datasets as tfds

    # Ratings data.
    ratings = tfds.load("movielens/100k-ratings", split="train")
    # Features of all the available movies.
    movies = tfds.load("movielens/100k-movies", split="train")

    ratings = ratings.map(
        lambda x: {
            "movie_title": x["movie_title"],
            "user_id": x["user_id"],
        }
    )
    movies = movies.map(lambda x: x["movie_title"])

    movies.save(movies_path, compression="GZIP")

    tf.random.set_seed(42)
    shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

    train = shuffled.take(80_000)
    test = shuffled.skip(80_000).take(20_000)

    train.save(train_path, compression="GZIP")
    test.save(test_path, compression="GZIP")

    movie_titles = movies.batch(1_000)
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    with open(file=unique_movie_titles_path, mode="wb") as f:
        pickle.dump(unique_movie_titles, f)
    with open(file=unique_user_ids_path, mode="wb") as f:
        pickle.dump(unique_user_ids, f)


prepare_dataset_op = kfp.components.create_component_from_func(
    prepare_dataset,
    output_component_file="prepare_dataset.yaml",
    base_image="eu.gcr.io/kubeflow-bg-experiment/recommender:latest",
)


def build_model(
    movies_path: comp.InputPath(),
    unique_user_ids_path: comp.InputPath(),
    unique_movie_titles_path: comp.InputPath(),
    train_path: comp.InputPath(),
    test_path: comp.InputPath(),
    gcs_bucket_name: str,
    model_version_number: str,
) -> None:
    """Function to build model."""
    import pickle
    from typing import Dict, Text

    import tensorflow as tf
    import tensorflow_recommenders as tfrs

    embedding_dimension = 32

    movies = tf.data.Dataset.load(movies_path, compression="GZIP")

    with open(file=unique_user_ids_path, mode="rb") as f:
        unique_user_ids = pickle.load(f)
    user_model = tf.keras.Sequential(
        [
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            # We add an additional embedding to account for unknown tokens.
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension),
        ]
    )

    with open(file=unique_movie_titles_path, mode="rb") as f:
        unique_movie_titles = pickle.load(f)
    movie_model = tf.keras.Sequential(
        [
            tf.keras.layers.StringLookup(
                vocabulary=unique_movie_titles, mask_token=None
            ),
            tf.keras.layers.Embedding(
                len(unique_movie_titles) + 1, embedding_dimension
            ),
        ]
    )
    metrics = tfrs.metrics.FactorizedTopK(candidates=movies.batch(128).map(movie_model))
    task = tfrs.tasks.Retrieval(metrics=metrics)

    class MovielensModel(tfrs.Model):
        def __init__(self, user_model, movie_model):
            super().__init__()
            self.movie_model: tf.keras.Model = movie_model
            self.user_model: tf.keras.Model = user_model
            self.task: tf.keras.layers.Layer = task

        def compute_loss(
            self, features: Dict[Text, tf.Tensor], training=False
        ) -> tf.Tensor:
            # We pick out the user features and pass them into the user model.
            user_embeddings = self.user_model(features["user_id"])
            # And pick out the movie features and pass them into the movie model,
            # getting embeddings back.
            positive_movie_embeddings = self.movie_model(features["movie_title"])

            # The task computes the loss and the metrics.
            return self.task(user_embeddings, positive_movie_embeddings)

    model = MovielensModel(user_model, movie_model)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    train = tf.data.Dataset.load(train_path, compression="GZIP")
    test = tf.data.Dataset.load(test_path, compression="GZIP")

    cached_train = train.shuffle(100_000).batch(8192).cache()
    cached_test = test.batch(4096).cache()

    model.fit(cached_train, epochs=3)
    model.evaluate(cached_test, return_dict=True)

    # Create a model that takes in raw query features, and
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    # recommends movies out of the entire movies dataset.
    index.index_from_dataset(
        tf.data.Dataset.zip(
            (movies.batch(100), movies.batch(100).map(model.movie_model))
        )
    )

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def serve(user_id):
        outputs = index(user_id)
        return outputs

    # model.save_weights(f"{model_path}", save_format="h5")
    tf.saved_model.save(
        index,
        f"gs://{gcs_bucket_name}/triton-recommender-retrieval/{model_version_number}/model.savedmodel/",
        signatures={"serving_default": serve.get_concrete_function()},
    )

build_model_op = kfp.components.create_component_from_func(
    build_model,
    output_component_file="build_model.yaml",
    base_image="eu.gcr.io/kubeflow-bg-experiment/recommender:latest",
)


# def trained_files_to_gcs(
#     # test_result: bool,
#     gcs_bucket_name: str,
#     model_path: comp.InputPath(),
# ) -> bool:
#     """Function to upload training output to gcs bucket."""
#     from google.cloud import storage

#     # if not test_result:
#     #     return

#     client = storage.Client()
#     bucket = client.bucket(gcs_bucket_name)

#     with open(file=f"{model_path}", mode="rb") as file:
#         blob = bucket.blob("basic_retrieval_model_weights.h5")
#         blob.upload_from_file(file, content_type="bytes")

#     return True


# trained_files_to_gcs_op = kfp.components.create_component_from_func(
#     trained_files_to_gcs,
#     output_component_file="trained_files_to_gcs.yaml",
#     base_image="eu.gcr.io/kubeflow-bg-experiment/recommender:latest",
# )


@dsl.pipeline(
    name="Basic retrieval pipeline",
    description="Basic retrieval pipeline",
)
def basic_retrieval_pipeline(gcs_bucket_name, model_version_number):
    """Function to run basic retrieval pipeline."""
    prepare_dataset_task = prepare_dataset_op()
    prepare_dataset_task.container.set_image_pull_policy("Always")
    prepare_dataset_task.set_caching_options(False)
    prepare_dataset_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    build_model_task = build_model_op(
        movies=prepare_dataset_task.outputs["movies"],
        unique_user_ids=prepare_dataset_task.outputs["unique_user_ids"],
        unique_movie_titles=prepare_dataset_task.outputs["unique_movie_titles"],
        train=prepare_dataset_task.outputs["train"],
        test=prepare_dataset_task.outputs["test"],
        gcs_bucket_name=gcs_bucket_name,
        model_version_number=model_version_number,
    )
    build_model_task.container.set_image_pull_policy("Always")
    build_model_task.set_caching_options(False)
    build_model_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    # trained_files_to_gcs_task = trained_files_to_gcs_op(
    #     gcs_bucket_name=gcs_bucket_name,
    #     model=build_model_task.outputs["model"],
    # )
    # trained_files_to_gcs_task.container.set_image_pull_policy("Always")
    # trained_files_to_gcs_task.set_caching_options(False)
    # trained_files_to_gcs_task.execution_options.caching_strategy.max_cache_staleness = (
    #     "P0D"
    # )


kfp.compiler.Compiler().compile(
    pipeline_func=basic_retrieval_pipeline, package_path="basic_retrieval_pipeline.yaml"
)

client = kfp.Client(host="http://localhost:3000")

# upload and run pipeline
pipeline_name = "basic retrieval pipeline"
pipeline_id = client.get_pipeline_id(pipeline_name)
if not pipeline_id:
    # create pipeline for the first time
    pipeline = client.upload_pipeline(
        pipeline_package_path="basic_retrieval_pipeline.yaml",
        pipeline_name=pipeline_name,
        description="basic retrieval pipeline",
    )
    pipeline_id = pipeline.id

# create a new version of the existing pipeline after deleteing all other versions
version_name = "latest"
versions = client.list_pipeline_versions(pipeline_id).versions
for version in versions:
    client.delete_pipeline_version(version.id)
version = client.upload_pipeline_version(
    pipeline_package_path="basic_retrieval_pipeline.yaml",
    pipeline_version_name=version_name,
    pipeline_id=pipeline_id,
)

experiment_name = "basic retrieval experiment"
try:
    experiment = client.get_experiment(experiment_name=experiment_name)
except:
    # create experiment for the first time
    experiment = client.create_experiment(name=experiment_name)
client.run_pipeline(
    experiment_id=experiment.id,
    job_name="basic retrieval job",
    params={
        "gcs_bucket_name": f"{os.environ.get('GCS_STORAGE_BUCKET_NAME')}",
        "model_version_number": f"{os.environ.get('MODEL_VERSION_NUMBER')}",
    },
    pipeline_id=pipeline_id,
    version_id=version.id,
)

# # recurring runs
# client.create_recurring_run(
#     experiment_id=experiment.id,
#     job_name="retrieval every hour",
#     interval_second=3600,
#     version_id=version.id,
# )

# # one off runs without pipeline and experiment
# client.create_run_from_pipeline_func(
#     basic_retrieval_pipeline,
#     arguments={
#         "gcs_bucket_name": f"{os.environ.get('GCS_STORAGE_BUCKET_NAME')}",
#     },
# )
