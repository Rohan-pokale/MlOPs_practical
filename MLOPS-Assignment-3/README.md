# MLOps Assignment 3 – Monitoring and Alerting

## What is it?
A Titanic Survival Prediction project that monitors model performance, data drift, anomalies, and API metrics using Prometheus and Grafana. It also includes automated alerting through Webhook.site.

## Project Structure

```text
MLOPS-Monitering/
├── data/                 # Dataset
├── models/               # Trained ML models
├── src/                  # Training and monitoring modules
├── prometheus/           # Prometheus configuration
├── main.py               # FastAPI application
├── docker-compose.yml    # Prometheus and Grafana services
├── pyproject.toml        # Project dependencies
├── uv.lock               # Dependency lock file
└── README.md
```

## Main Components

- **Machine Learning:** Logistic Regression, Decision Tree, Random Forest.
- **FastAPI:** Serves the selected Decision Tree model.
- **Prometheus:** Collects model and API metrics.
- **Grafana:** Visualizes metrics and configures alerts.
- **Drift Detection:** Uses Population Stability Index (PSI).
- **Anomaly Detection:** Uses Isolation Forest.
- **Webhook.site:** Receives alert notifications.

## How to Run

**1. Install dependencies**

```bash
uv sync
```

**2. Start Prometheus and Grafana**

Make sure Docker Desktop is running.

```bash
docker compose up -d
```

**3. Start the FastAPI application**

```bash
uv run uvicorn main:app --reload
```

## Access Services

| Service | URL |
|---|---|
| FastAPI Swagger | http://127.0.0.1:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| Metrics | http://127.0.0.1:8000/metrics |

**Stop Docker services:**

```bash
docker compose down
```