from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from app.api.routes import router as api_router
from app.core.config import settings
from app.core.logging import get_logger
from app.services.storage import storage
from app.ml.model import AnomalyDetector

logger = get_logger(__name__)

app = FastAPI(title=settings.APP_NAME)
app.include_router(api_router, prefix="")

# initialize model singleton
model = AnomalyDetector(settings.MODEL_PATH)

@app.on_event("startup")
def startup_event():
    logger.info("Starting Observa Sentinel")
    # Attempt to load model if exists
    try:
        model.load()
        logger.info("Model loaded at startup (if available)")
    except Exception as exc:
        logger.warning(f"Model not loaded: {exc}")

@app.get("/metrics")
def metrics():
    """Expose prometheus metrics"""
    data = generate_latest()
    return JSONResponse(content=data.decode('utf-8'), media_type=CONTENT_TYPE_LATEST)
