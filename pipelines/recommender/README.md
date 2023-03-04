# Recommender pipeline
References:
- https://www.tensorflow.org/recommenders/examples/basic_retrieval
- https://www.tensorflow.org/recommenders/examples/basic_ranking
```shell
# in a separate terminal, port forward from kubeflow pipeline ui service
kubectl port-forward --namespace kubeflow svc/ml-pipeline-ui 3000:80

# build pipeline dependencies docker image
cd pipelines/recommender
docker build -t eu.gcr.io/<PROJECT ID>/recommender:latest .
# example,
docker build -t eu.gcr.io/kubeflow-bg-experiment/recommender:latest .

# push docker image
gcloud auth configure-docker
docker push eu.gcr.io/<PROJECT ID>/recommender:latest
# example,
docker push eu.gcr.io/kubeflow-bg-experiment/recommender:latest

# install kubeflow pip dependencies
pip install kfp --upgrade

# install dependencies
pip install -r requirements.txt

# run the basic retrieval pipeline
GCS_STORAGE_BUCKET_NAME="kubeflow-prototype-storagebucket01" \
  python retrieval/basic_retrieval.py

# run the listwise ranking pipeline
GCS_STORAGE_BUCKET_NAME="kubeflow-prototype-storagebucket01" \
  python ranking/listwise_ranking.py
```
