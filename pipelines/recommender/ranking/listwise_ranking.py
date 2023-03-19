import os

import kfp
import kfp.components as comp
import kfp.dsl as dsl


def prepare_dataset(
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
    import tensorflow_recommenders as tfrs

    ratings = tfds.load("movielens/100k-ratings", split="train")
    movies = tfds.load("movielens/100k-movies", split="train")

    ratings = ratings.map(
        lambda x: {
            "movie_title": x["movie_title"],
            "user_id": x["user_id"],
            "user_rating": x["user_rating"],
        }
    )
    movies = movies.map(lambda x: x["movie_title"])

    unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))
    unique_user_ids = np.unique(
        np.concatenate(list(ratings.batch(1_000).map(lambda x: x["user_id"])))
    )

    with open(file=unique_movie_titles_path, mode="wb") as f:
        pickle.dump(unique_movie_titles, f)
    with open(file=unique_user_ids_path, mode="wb") as f:
        pickle.dump(unique_user_ids, f)

    tf.random.set_seed(42)

    # Split between train and tests sets, as before.
    shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

    train = shuffled.take(80_000)
    test = shuffled.skip(80_000).take(20_000)

    # We sample 50 lists for each user for the training data. For each list we
    # sample 5 movies from the movies the user rated.
    train = tfrs.examples.movielens.sample_listwise(
        train, num_list_per_user=50, num_examples_per_list=5, seed=42
    )
    test = tfrs.examples.movielens.sample_listwise(
        test, num_list_per_user=1, num_examples_per_list=5, seed=42
    )

    train.save(train_path, compression="GZIP")
    test.save(test_path, compression="GZIP")


prepare_dataset_op = kfp.components.create_component_from_func(
    prepare_dataset,
    output_component_file="prepare_dataset.yaml",
    base_image="eu.gcr.io/kubeflow-bg-experiment/recommender:latest",
)


def build_model(
    unique_user_ids_path: comp.InputPath(),
    unique_movie_titles_path: comp.InputPath(),
    train_path: comp.InputPath(),
    test_path: comp.InputPath(),
    gcs_bucket_name: str,
) -> None:
    """Function to build model."""
    import pickle

    import tensorflow as tf
    import tensorflow_ranking as tfr
    import tensorflow_recommenders as tfrs

    with open(file=unique_user_ids_path, mode="rb") as f:
        unique_user_ids = pickle.load(f)
    with open(file=unique_movie_titles_path, mode="rb") as f:
        unique_movie_titles = pickle.load(f)

    class RankingModel(tfrs.Model):
        def __init__(self, loss):
            super().__init__()
            embedding_dimension = 32

            # Compute embeddings for users.
            self.user_embeddings = tf.keras.Sequential(
                [
                    tf.keras.layers.StringLookup(vocabulary=unique_user_ids),
                    tf.keras.layers.Embedding(
                        len(unique_user_ids) + 2, embedding_dimension
                    ),
                ]
            )

            # Compute embeddings for movies.
            self.movie_embeddings = tf.keras.Sequential(
                [
                    tf.keras.layers.StringLookup(vocabulary=unique_movie_titles),
                    tf.keras.layers.Embedding(
                        len(unique_movie_titles) + 2, embedding_dimension
                    ),
                ]
            )

            # Compute predictions.
            self.score_model = tf.keras.Sequential(
                [
                    # Learn multiple dense layers.
                    tf.keras.layers.Dense(256, activation="relu"),
                    tf.keras.layers.Dense(64, activation="relu"),
                    # Make rating predictions in the final layer.
                    tf.keras.layers.Dense(1),
                ]
            )

            self.task = tfrs.tasks.Ranking(
                loss=loss,
                metrics=[
                    tfr.keras.metrics.NDCGMetric(name="ndcg_metric"),
                    tf.keras.metrics.RootMeanSquaredError(),
                ],
            )

        def call(self, features):
            # We first convert the id features into embeddings.
            # User embeddings are a [batch_size, embedding_dim] tensor.
            user_embeddings = self.user_embeddings(features["user_id"])

            # Movie embeddings are a [batch_size, num_movies_in_list, embedding_dim]
            # tensor.
            movie_embeddings = self.movie_embeddings(features["movie_title"])

            # We want to concatenate user embeddings with movie emebeddings to pass
            # them into the ranking model. To do so, we need to reshape the user
            # embeddings to match the shape of movie embeddings.
            list_length = features["movie_title"].shape[1]
            user_embedding_repeated = tf.repeat(
                tf.expand_dims(user_embeddings, 1), [list_length], axis=1
            )

            # Once reshaped, we concatenate and pass into the dense layers to generate
            # predictions.
            concatenated_embeddings = tf.concat(
                [user_embedding_repeated, movie_embeddings], 2
            )

            return self.score_model(concatenated_embeddings)

        def compute_loss(self, features, training=False):
            labels = features.pop("user_rating")

            scores = self(features)

            return self.task(
                labels=labels,
                predictions=tf.squeeze(scores, axis=-1),
            )

    train = tf.data.Dataset.load(train_path, compression="GZIP")
    test = tf.data.Dataset.load(test_path, compression="GZIP")

    cached_train = train.shuffle(100_000).batch(8192).cache()
    cached_test = test.batch(4096).cache()

    listwise_model = RankingModel(tfr.keras.losses.ListMLELoss())
    listwise_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

    epochs = 1
    listwise_model.fit(cached_train, epochs=epochs, verbose=False)

    # listwise_model.save_weights(f"{model_path}", save_format="h5")
    tf.saved_model.save(
        listwise_model,
        f"gs://{gcs_bucket_name}/listwise-ranking-model/1/",
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
#         blob = bucket.blob("listwise_ranking_model_weights.h5")
#         blob.upload_from_file(file, content_type="bytes")

#     return True


# trained_files_to_gcs_op = kfp.components.create_component_from_func(
#     trained_files_to_gcs,
#     output_component_file="trained_files_to_gcs.yaml",
#     base_image="eu.gcr.io/kubeflow-bg-experiment/recommender:latest",
# )


@dsl.pipeline(
    name="Listwise ranking pipeline",
    description="Listwise ranking pipeline",
)
def listwise_ranking_pipeline(gcs_bucket_name):
    """Function to run listwise ranking pipeline."""
    prepare_dataset_task = prepare_dataset_op()
    prepare_dataset_task.container.set_image_pull_policy("Always")
    prepare_dataset_task.set_caching_options(False)
    prepare_dataset_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    build_model_task = build_model_op(
        unique_user_ids=prepare_dataset_task.outputs["unique_user_ids"],
        unique_movie_titles=prepare_dataset_task.outputs["unique_movie_titles"],
        train=prepare_dataset_task.outputs["train"],
        test=prepare_dataset_task.outputs["test"],
        gcs_bucket_name=gcs_bucket_name,
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
    pipeline_func=listwise_ranking_pipeline,
    package_path="listwise_ranking_pipeline.yaml",
)

client = kfp.Client(host="http://localhost:3000")

# upload and run pipeline
pipeline_name = "listwise ranking pipeline"
pipeline_id = client.get_pipeline_id(pipeline_name)
if not pipeline_id:
    # create pipeline for the first time
    pipeline = client.upload_pipeline(
        pipeline_package_path="listwise_ranking_pipeline.yaml",
        pipeline_name=pipeline_name,
        description="listwise ranking pipeline",
    )
    pipeline_id = pipeline.id

# create a new version of the existing pipeline after deleteing all other versions
version_name = "latest"
versions = client.list_pipeline_versions(pipeline_id).versions
for version in versions:
    client.delete_pipeline_version(version.id)
version = client.upload_pipeline_version(
    pipeline_package_path="listwise_ranking_pipeline.yaml",
    pipeline_version_name=version_name,
    pipeline_id=pipeline_id,
)

experiment_name = "listwise ranking experiment"
try:
    experiment = client.get_experiment(experiment_name=experiment_name)
except:
    # create experiment for the first time
    experiment = client.create_experiment(name=experiment_name)
client.run_pipeline(
    experiment_id=experiment.id,
    job_name="listwise ranking job",
    params={
        "gcs_bucket_name": f"{os.environ.get('GCS_STORAGE_BUCKET_NAME')}",
    },
    pipeline_id=pipeline_id,
    version_id=version.id,
)

# # recurring runs
# client.create_recurring_run(
#     experiment_id=experiment.id,
#     job_name="ranking every hour",
#     interval_second=3600,
#     version_id=version.id,
# )

# # one off runs without pipeline and experiment
# client.create_run_from_pipeline_func(
#     listwise_ranking_pipeline,
#     arguments={
#         "gcs_bucket_name": f"{os.environ.get('GCS_STORAGE_BUCKET_NAME')}",
#     },
# )
