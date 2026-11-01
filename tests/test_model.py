import os
import tempfile
import numpy as np
from app.ml.model import AnomalyDetector

def test_train_and_predict(tmp_path):
    # create synthetic sequences
    seqs = []
    for i in range(6):
        t = np.linspace(0, 6, 100)
        vals = np.sin(t * (1 + i*0.1)) + np.random.normal(scale=0.01, size=t.shape)
        seqs.append(vals.tolist())

    out = tmp_path / 'model.pt'
    det = AnomalyDetector(str(out))
    det.train(seqs, epochs=2)
    det.save()
    assert out.exists()

    det2 = AnomalyDetector(str(out))
    det2.load()
    # create a normal window and an anomalous window
    normal = seqs[0][-det2.window:]
    anomaly = normal.copy()
    anomaly[-1] += 10.0  # spike
    score_normal, is_anom_normal = det2.predict(normal)
    score_anom, is_anom = det2.predict(anomaly)
    assert score_anom >= score_normal
