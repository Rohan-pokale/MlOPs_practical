# MLOps Kubernetes — Titanic Survival Prediction

This project demonstrates deploying a Machine Learning model as a scalable and resilient service using Kubernetes.

## Tech Stack

* Python
* uv
* FastAPI
* Scikit-learn
* Docker
* Kubernetes
* Minikube
* kubectl

## Workflow

```text
Titanic ML Model
      ↓
FastAPI
      ↓
Docker
      ↓
Kubernetes Deployment
      ↓
Kubernetes Service
      ↓
Scaling & Resilience
```

## Kubernetes Experiments

* Deployed the Titanic Random Forest model using Kubernetes.
* Exposed the model through a Kubernetes NodePort Service.
* Scaled the deployment from 1 to 3 replicas.
* Deleted a running Pod and verified Kubernetes automatically created a replacement.

## Run

Start Minikube:

```bash
minikube start --driver=docker
```

Deploy:

```bash
minikube kubectl -- apply -f k8s/deployment.yaml
minikube kubectl -- apply -f k8s/service.yaml
```

Check:

```bash
minikube kubectl -- get pods
minikube kubectl -- get services
```

Access the API:

```bash
minikube service titanic-ml-service --url
```

API documentation is available at `/docs`.

## Result

The ML model was successfully deployed as a Kubernetes service and tested for scaling and self-healing.
