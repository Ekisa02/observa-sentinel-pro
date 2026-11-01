from fastapi import APIRouter, Body, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Optional
from app.services.storage import storage
from app.ml.model import AnomalyDetector
from app.core.config import settings
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

class IngestPayload(BaseModel):
    series_id: str
    points: List[Tuple[float, float]]  # list of [timestamp, value]

class AnalyzePayload(BaseModel):
    series_id: Optional[str] = None
    points: Optional[List[Tuple[float, float]]] = None

# model singleton imported from app.main to avoid reinitialization in tests
from app.main import model  # noqa: E402

@router.post("/ingest")
def ingest(payload: IngestPayload):
    storage.add_series(payload.series_id, payload.points)
    logger.debug(f"Ingested series {payload.series_id} with {len(payload.points)} points")
    return {"status": "ok", "series_id": payload.series_id, "count": len(payload.points)}

@router.post("/analyze")
def analyze(payload: AnalyzePayload = Body(...)):
    if payload.points is None:
        if not payload.series_id:
            raise HTTPException(status_code=400, detail="either points or series_id must be provided")
        points = storage.get_series(payload.series_id)
        if not points:
            raise HTTPException(status_code=404, detail="series not found")
    else:
        points = payload.points

    # extract values
    values = [float(p[1]) for p in points]
    result = model.predict(values)
    return {"anomaly_score": result[0], "is_anomaly": bool(result[1])}

@router.post("/train")
def train(background_tasks: BackgroundTasks):
    """Trigger a training task in background using all stored series (demo)."""
    def _train_task():
        logger.info("Background training started")
        data = storage.all_series_values()
        if not data:
            logger.warning("No data available for training")
            return
        model.train(data)
        model.save()
        logger.info("Background training finished")

    background_tasks.add_task(_train_task)
    return {"status": "training_started"}
