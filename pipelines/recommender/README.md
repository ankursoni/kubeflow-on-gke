# Recommender pipeline


## Pre-requisite
References:
- https://istio.io/latest/docs/setup/platform-setup/gke/
- https://istio.io/latest/docs/setup/install/helm/
- https://istio.io/latest/docs/tasks/traffic-management/ingress/ingress-control/
- https://www.kubeflow.org/docs/external-add-ons/serving/seldon/#seldon-serving
<!-- # seldon v1 -->
<!-- - https://docs.seldon.io/projects/seldon-core/en/latest/workflow/install.html
- https://docs.seldon.io/projects/seldon-core/en/latest/ingress/istio.html
- https://docs.seldon.io/projects/seldon-core/en/latest/graph/inference-graph.html
- https://docs.seldon.io/projects/seldon-core/en/latest/servers/overview.html#example-for-gcp-gke
- https://docs.seldon.io/projects/seldon-core/en/latest/examples/graph-metadata.html#Two-Level-Graph -->
<!-- - # seldon v2 -->
- https://docs.seldon.io/projects/seldon-core/en/v2/contents/getting-started/kubernetes-installation/helm.html
```shell
# list firewall rules for master node
gcloud compute firewall-rules list --filter="name~gke-<CLUSTER_NAME>-[0-9a-z]*-master"
# example,
gcloud compute firewall-rules list --filter="name~gke-kubeflow-prototype-gke01-[0-9a-z]*-master"

# add firewall rule for port 15017 (istio) and 4443 (seldon)
gcloud compute firewall-rules update <FIREWALL_RULE_NAME> --allow tcp:10250,tcp:443,tcp:15017,tcp:4443
# example,
gcloud compute firewall-rules update gke-kubeflow-prototype-gke01-74e391f2-master --allow tcp:10250,tcp:443,tcp:15017,tcp:4443

# install istio
helm repo add istio https://istio-release.storage.googleapis.com/charts
helm repo update

kubectl create namespace istio-system
helm upgrade -i istio-base istio/base -n istio-system
helm ls -n istio-system
helm status istio-base -n istio-system

helm upgrade -i istiod istio/istiod -n istio-system --wait
helm ls -n istio-system
helm status istiod -n istio-system

kubectl get deployments -n istio-system --output wide

kubectl create namespace istio-ingress
helm upgrade -i istio-ingress istio/gateway -n istio-ingress --wait
helm ls -n istio-ingress
helm status istio-ingress -n istio-ingress

# install seldon

# # v1
# helm repo add seldonio https://storage.googleapis.com/seldon-charts
# helm repo update

# kubectl create namespace seldon-system
# helm upgrade -i seldon-core-operator seldonio/seldon-core-operator \
#     --set usageMetrics.enabled=true \
#     --set istio.enabled=true \
#     --namespace seldon-system
# helm ls -n seldon-system
# helm status seldon-core-operator -n seldon-system

# kubectl create namespace seldon
# kubectl label namespace seldon serving.kubeflow.org/inferenceservice=enabled

# v2
helm repo add seldon-charts https://seldonio.github.io/helm-charts
helm repo update seldon-charts

helm install seldon-core-v2-crds seldon-charts/seldon-core-v2-crds

kubectl create namespace seldon-mesh
helm install seldon-core-v2 seldon-charts/seldon-core-v2-setup --namespace seldon-mesh
helm install seldon-v2-servers seldon-charts/seldon-core-v2-servers --namespace seldon-mesh

# install seldon gateway running on port 80
cd pipelines/recommender
kubectl apply -f seldon_gateway.yaml

# run simple example on seldon
kubectl apply -f seldon_simple_example.yaml
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
MODEL_VERSION_NUMBER="1" \
  python basic_retrieval.py

# run the listwise ranking pipeline
cd ../ranking
GCS_STORAGE_BUCKET_NAME="kubeflow-prototype-storagebucket01" \
MODEL_VERSION_NUMBER="1" \
  python listwise_ranking.py


# after model output is saved to gcs bucket then,
# run listwise ranking model serve on seldon
cd ..
kubectl apply -f seldon_listwise_ranking_model_serve.yaml
```
