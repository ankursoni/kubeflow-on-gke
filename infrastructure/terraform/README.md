This terraform deploys:

- A VPC with one subnet and 2 secondary ip ranges meant for kubernetes pods and services.
- A Google Cloud Storage (GCS) bucket.
- A Google Container Registry (GCR) in EU region.
- A standard Google Kubernetes Engine (GKE) cluster from RAPID release channel with:
  - 3 node pools:
    - nodepool01 is meant to contain standard AMD64 machine type VMs,
    - nodepool02 is meant to contain new ARM64 machine type VMs and,
    - nodepool03 is meant to contain SPOT VMs.
  - A common Google Service Account (GSA) for GKE node pools with access to GCR and GCS bucket.
  - 2 GSAs for kubeflow kfp-system and kfp-user with read and write access to GCS bucket respectively.


# Login to gcloud
```shell
# login via web
gcloud auth login

# list gcp projects
gcloud projects list

# set gcp project in config
gcloud config set project <PROJECT ID>
# example,
gcloud config set project kubeflow-bg-experiment
```


# Setup GKE

## References
ARM based T2A series VMs are available in select regions
- https://cloud.google.com/blog/products/compute/tau-t2a-is-first-compute-engine-vm-on-an-arm-chip#:~:text=T2A%20VMs%20are%20generally%20available,Availability%20in%20the%20coming%20months.
```shell
# get regions list
gcloud compute regions list

# get supported gke versions in RAPID channel
gcloud container get-server-config --flatten="channels" --filter="channels.channel=RAPID" \
    --format="yaml(channels.channel,channels.defaultVersion)" --region europe-west4
```

## Create secret vars file
```shell
cd infrastructure/terraform

cp values.tfvars values-secret.tfvars

# then, modify the values in values-secret.tfvars
```

## Enable GCP APIs
https://googlecloudplatform.github.io/kubeflow-gke-docs/docs/deploy/project-setup/
```shell
gcloud services enable \
  serviceusage.googleapis.com \
  compute.googleapis.com \
  container.googleapis.com \
  iam.googleapis.com \
  servicemanagement.googleapis.com \
  cloudresourcemanager.googleapis.com \
  ml.googleapis.com \
  iap.googleapis.com \
  sqladmin.googleapis.com \
  meshconfig.googleapis.com \
  krmapihosting.googleapis.com \
  servicecontrol.googleapis.com \
  endpoints.googleapis.com \
  iamcredentials.googleapis.com
```

## Apply terraform
```shell
cd infrastructure/terraform

terraform init

terraform apply -var-file="values-secret.tfvars"
```

## Get GKE credentials for kube context
```shell
gcloud container clusters get-credentials <PREFIX>-<ENVIRONMENT>-gke01 --region europe-west4 --project <PROJECT ID>
# example,
gcloud container clusters get-credentials kubeflow-prototype-gke01 --region europe-west4 --project kubeflow-bg-experiment
```

## Destroy terraform
```shell
terraform destroy -var-file="values-secret.tfvars"
```


# Install Kubeflow

## Install Kubeflow v1
Installation:
- https://www.kubeflow.org/docs/components/pipelines/v1/installation/standalone-deployment/
```shell
export PIPELINE_VERSION=1.8.5
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"
```

## Authenticating Kubeflow to Google Cloud
Setup authentication pipelines:
- https://v1-6-branch.kubeflow.org/docs/distributions/gke/authentication/

```shell
export PROJECT_ID="<PROJECT ID>" # example, kubeflow-bg-experiment
export NAMESPACE="kubeflow"
export GSA="gke01-kfp-user@${PROJECT_ID}.iam.gserviceaccount.com"
export KSA="pipeline-runner"

# annotate kubernetes service account to use google service account (already provisioned by terraform) that has iam role to read/write to google cloud storage bucket
kubectl annotate serviceaccount \
  --namespace $NAMESPACE \
  --overwrite \
  $KSA \
  iam.gke.io/gcp-service-account=$GSA

# run the above for the following combination as well
export GSA="gke01-kfp-system@${PROJECT_ID}.iam.gserviceaccount.com"
export KSA="ml-pipeline-ui"

export GSA="gke01-kfp-system@${PROJECT_ID}.iam.gserviceaccount.com"
export KSA="ml-pipeline-visualizationserver"

# port forward from kubeflow pipeline ui service
kubectl port-forward --namespace kubeflow svc/ml-pipeline-ui 3000:80
```

## Cloud IAP
Setup Cloud IAP:
- https://googlecloudplatform.github.io/kubeflow-gke-docs/docs/deploy/oauth-setup/
- https://cloud.google.com/iap/docs/
```shell
TODO:
```

## Delete Kubeflow v1
```shell
export PIPELINE_VERSION=1.8.5
kubectl delete -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"
kubectl delete -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
```

Now, check [pipelines/hello_world/README](../../pipelines/hello_world/README.md)
