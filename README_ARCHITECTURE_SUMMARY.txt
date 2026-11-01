Components:
- API: FastAPI with modular routing
- Storage: simple memory-backed store + CSV flush (demo)
- ML: PyTorch Autoencoder for anomaly detection
- Telemetry: Prometheus metrics

Interaction:
Clients -> FastAPI (ingest) -> Storage -> ML (analyze/train)
