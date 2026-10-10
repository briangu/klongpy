# Ticker Plant Terraform Deployment

This Terraform module deploys the pub-sub ticker plant example `srv_pubsub.kg` to Kubernetes. The script is stored in a `ConfigMap` and executed using the `klongpy-app:latest` image from the Docker example.

## Usage

Ensure your Kubernetes context is configured and the `klongpy-app:latest` image is accessible by the cluster. Then run:

```bash
terraform init
terraform apply
```

The deployment creates the `klongpy` namespace, a `Deployment` running the ticker plant, and a `Service` exposing port `8888`.
