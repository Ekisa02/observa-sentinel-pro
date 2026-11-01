from typing import List, Tuple
from app.services.storage import storage

def ingest_series(series_id: str, points: List[Tuple[float, float]]):
    storage.add_series(series_id, points)
    return True
