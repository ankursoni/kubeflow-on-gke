# Hello world pipeline
References:
- https://www.kubeflow.org/docs/components/pipelines/v1/sdk/install-sdk/
```shell
# in a separate terminal, port forward from kubeflow pipeline ui service
kubectl port-forward --namespace kubeflow svc/ml-pipeline-ui 3000:80

# build pipeline dependencies docker image
cd pipelines/hello_world
docker build -t eu.gcr.io/<PREFIX>-<ENVIRONMENT>/hello_world:latest .
# example,
docker build -t eu.gcr.io/kubeflow-on-gke/hello_world:latest .

# push docker image
gcloud auth configure-docker
docker push eu.gcr.io/<PREFIX>-<ENVIRONMENT>/hello_world:latest
# example,
docker push eu.gcr.io/kubeflow-on-gke/hello_world:latest

# install kubeflow pip dependencies
pip install kfp --upgrade

# run the hello world pipeline
USER_NAME="Kubeflow User" \
GCS_STORAGE_BUCKET_NAME="kubeflow-prototype-storagebucket01" \
FILE_NAME="hello_message.txt" \
  python hello_world.py
```
