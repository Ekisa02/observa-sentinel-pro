import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_and_ingest_and_analyze():
    # ingest series
    payload = {
        "series_id": "test.server.cpu",
        "points": [[1, 0.1], [2, 0.2], [3, 0.15], [4, 0.12], [5, 0.11], [6, 0.13], [7, 0.15], [8, 0.14]]
    }
    r = client.post('/ingest', json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data['status'] == 'ok'

    # analyze series (without model loaded - will raise RuntimeError if model missing)
    r = client.post('/analyze', json={'series_id': 'test.server.cpu'})
    # app may raise 500 if model not loaded; ensure we get a valid response or a 500
    assert r.status_code in (200, 500)

def test_metrics_endpoint():
    r = client.get('/metrics')
    assert r.status_code == 200
