from typing import List
from app.ml.model import AnomalyDetector

def train_from_sequences(sequences: List[List[float]], output_path: str, epochs: int = 10):
    det = AnomalyDetector(output_path)
    det.train(sequences, epochs=epochs)
    det.save(output_path)
    return output_path
