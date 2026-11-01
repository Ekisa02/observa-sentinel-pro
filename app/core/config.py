from pydantic import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Observa Sentinel"
    IN_MEMORY_LIMIT: int = 10000
    MODEL_PATH: str = "models/autoencoder.pt"
    PROMETHEUS_PREFIX: str = "observasentinel"

    class Config:
        env_prefix = "OBS_"

settings = Settings()
