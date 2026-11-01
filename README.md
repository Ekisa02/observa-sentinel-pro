# Observa Sentinel

Observa Sentinel is a modular DevOps observability platform built with FastAPI and PyTorch that provides real-time metric ingestion, storage, and anomaly detection using lightweight ML models. It is designed as a minimal, production-oriented reference implementation that demonstrates core architecture patterns for observability platforms: ingest, store, analyze, and expose APIs and telemetry.

Key features
- FastAPI-based REST API for ingestion and analysis
- PyTorch-based autoencoder model for time-series anomaly detection
- Lightweight in-memory + disk storage abstraction
- Prometheus metrics endpoint integration
- Training and evaluation scripts
- Tests and documentation

Architecture

The repository follows a modular architecture:

- app/
  - main.py: FastAPI application, routes and application startup
  - api/: HTTP route handlers and pydantic schemas
  - ml/: PyTorch model and training utilities
  - services/: ingestion/storage helpers
  - core/: configuration and logging utilities
- scripts/: CLI helpers for training and running the server
- tests/: unit/integration tests
- config/: runtime configuration and example configs
- docs/: architecture and usage notes

A simple ASCII diagram:

    +------------+      +-------------+      +------------+
    |  Clients   | ---> | FastAPI API | ---> | In-memory  |
    |  (push)    |      | (endpoints) |      | Storage    |
    +------------+      +-------------+      +------------+
                                       |         |
                                       v         v
                                 +-----------+  +------------+
                                 |  ML Model |  | Persistent |
                                 | (PyTorch) |  |  Storage   |
                                 +-----------+  +------------+
                                       |
                                       v
                                /metrics (Prometheus)

Installation

Requirements
- Python 3.9+
- Git

Quick start (virtualenv)

1. Clone

   git clone https://example.com/observa-sentinel.git
   cd observa-sentinel

2. Create venv and install

   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

3. Run the server (development)

   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

4. Train a sample model (optional)

   python scripts/train_model.py --epochs 3 --output models/autoencoder.pt

API - Key endpoints

- POST /ingest
  - Accepts a JSON payload containing metric series (timestamp, value).
  - Example:
    {
      "series_id": "server-1.cpu",
      "points": [[1620000000, 0.12], [1620000060, 0.09]]
    }

- POST /analyze
  - Runs anomaly detection on a provided series or on stored series.
  - Example payload:
    {
      "series_id": "server-1.cpu",
      "points": [[...]]
    }

- POST /train
  - Trains a model server-side using stored data (demo endpoint).

- GET /metrics
  - Prometheus-compatible metrics exposed for scraping.

Configuration

See config/config.yaml for sample runtime options. The app uses pydantic-based configuration defined in app/core/config.py.

Development and testing

Run tests with pytest:

   pytest -q

Files of interest
- app/main.py - application bootstrap
- app/ml/model.py - PyTorch model definition and utilities
- scripts/train_model.py - training CLI using app/ml/trainer.py
- tests/ - unit and integration tests

Contributing

This is a reference project. Contributions are welcome; please open issues or PRs for improvements.
