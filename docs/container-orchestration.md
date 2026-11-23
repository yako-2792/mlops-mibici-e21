# Container Orchestration

## Overview
---
This document describes the orchestration of the core services for the MiBici project using Docker Compose (`compose.yml`). Its goal is to launch and manage both the MLflow experiment tracking server (`mibici-mlflow`) and the FastAPI inference service (`mibici-api`) as independent microservices within a single stack, enabling reproducible local execution and deployment without manual step-by-step setup.   

## How It Works
---
Based on the definitions in `compose.yml`, Docker Compose builds the API image from the local `Containerfile` and pulls the official MLflow image, running both containers inside an isolated internal network. The `mibici-mlflow` service (port 5000) mounts the local SQLite database (`mlflow.db`) and artifacts directory (`mlartifacts/`) to preserve experiment history, while `mibici-api` (port 8000) starts after MLflow and serves prediction endpoints using the exported model in `models/p1-demanda/`.

## Commands
---
Use the following commands from the project root to manage the container stack:

* **`docker compose up -d`**
  Starts all services (`mibici-mlflow` and `mibici-api`) in detached mode (background) using existing images.

* **`docker compose up -d --build`**
  Rebuilds the API image from `Containerfile` (use after modifying `api/` code or updating `models/p1-demanda/`) and starts the stack in the background.

* **`docker compose ps`**
  Displays the current status, health checks, and port mappings of the running services.

* **`docker compose logs -f`**
  Streams real-time logs from all services (append `api` or `mlflow` to filter a specific container; press `Ctrl + C` to exit without stopping the stack).

* **`docker compose restart`**
  Restarts the running containers without rebuilding images (useful after updating environment variables in `compose.yml`).

* **`docker compose stop`**
  Stops the running containers without removing them, allowing a faster restart later with `docker compose start`.

* **`docker compose down`**
  Stops and removes the containers and internal network while keeping local persistence files (`mlflow.db` and `mlartifacts/`) intact.

### Quick Check

Once the stack is running, verify that both services are healthy and reachable:

1. **Check container status:**
   ```bash
   docker compose ps
   ```

2. **Check API health endpoint**
    ```bash
    curl http://localhost:8000/health
    ```

3. **Verify MLFlow UI**
    * MLFlow Tracking UI: `http://localhost:5000`

## How to Use
---

Depending on your local workspace state, follow one of these two startup scenarios: 

* **Scenario A:** Existing `mlflow.db` and `models/` (Ready to run)
* **Scenario B:** Fresh clone (No local database or model)

### Scenario A

If `mlflow.db`, `mlartifacts/`, and `models/p1-demanda/` are already present in your workspace, build and start the stack directly:

```bash
# Build and start the stack in the background
docker compose up -d --build
```

### Scenario B

If `mlflow.db` and `models/` do not exist yet in your workspace, generate them before starting the containers (otherwise Docker will mount an empty directory in place of `mlflow.db` and the API will fail to load the model):

1. Ingest training data and materialize gold features.

```bash
make ingest
```

2. Train a P1 candidate model, and register champion.

```bash
make train Q=p1 MODEL=rf
make register Q=p1
```

3. Export the champion model to `models/p1-demanda/`.

```bash
make export-model Q=p1
```

4. Build the API image and start the stack in the background.

```bash
docker compose up -d --build
```
