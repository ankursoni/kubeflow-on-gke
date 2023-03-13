# Recommender pipeline


## Pre-requisite
References:
- https://istio.io/latest/docs/setup/platform-setup/gke/
- https://istio.io/latest/docs/setup/install/helm/
- https://istio.io/latest/docs/tasks/traffic-management/ingress/ingress-control/
- https://docs.seldon.io/projects/seldon-core/en/latest/workflow/install.html
- https://docs.seldon.io/projects/seldon-core/en/latest/ingress/istio.html
- https://www.kubeflow.org/docs/external-add-ons/serving/seldon/#seldon-serving
```shell
# list firewall rules for master node
gcloud compute firewall-rules list --filter="name~gke-<CLUSTER_NAME>-[0-9a-z]*-master"
#example
gcloud compute firewall-rules list --filter="name~gke-kubeflow-prototype-gke01-[0-9a-z]*-master"

# add firewall rule for port 15017
gcloud compute firewall-rules update <FIREWALL_RULE_NAME> --allow tcp:10250,tcp:443,tcp:15017
# example,
gcloud compute firewall-rules update gke-kubeflow-prototype-gke01-74e391f2-master --allow tcp:10250,tcp:443,tcp:15017

# install istio
helm repo add istio https://istio-release.storage.googleapis.com/charts
helm repo update

kubectl create namespace istio-system
helm install istio-base istio/base -n istio-system
helm ls -n istio-system

helm install istiod istio/istiod -n istio-system --wait
helm ls -n istio-system
helm status istiod -n istio-system

kubectl get deployments -n istio-system --output wide

kubectl create namespace istio-ingress
helm install istio-ingress istio/gateway -n istio-ingress --wait
helm status istio-ingress -n istio-ingress

# install seldon
kubectl create namespace seldon-system
helm install seldon-core seldon-core-operator \
    --repo https://storage.googleapis.com/seldon-charts \
    --set usageMetrics.enabled=true \
    --set istio.enabled=true \
    --namespace seldon-system

kubectl create namespace seldon
kubectl label namespace seldon serving.kubeflow.org/inferenceservice=enabled

# install seldon gateway running on port 80
cd pipelines/recommender
kubectl create -f seldon_gateway.yaml

# run simple example on seldon
kubectl create -n seldon -f seldon_simple_example.yaml
```


## Run pipeline
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

# run the basic retrieval pipeline
cd retrieval
GCS_STORAGE_BUCKET_NAME="kubeflow-prototype-storagebucket01" \
  python basic_retrieval.py

# run the listwise ranking pipeline
cd ../ranking
GCS_STORAGE_BUCKET_NAME="kubeflow-prototype-storagebucket01" \
  python listwise_ranking.py
```
