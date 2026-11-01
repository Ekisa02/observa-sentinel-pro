# Architecture Overview

Observa Sentinel uses a small set of components to model an observability pipeline:

- API layer (FastAPI): Accepts ingestion requests and client queries.
- Storage layer: Lightweight in-memory buffer with optional disk persistence (CSV) for demo purposes.
- ML layer (PyTorch): Autoencoder-based detector for metric anomalies.
- Telemetry: Prometheus metrics exported via prometheus_client.

The implementation aims to be small but idiomatic, showing separation of concerns and testability.
