from pydantic import BaseModel
from typing import List, Tuple

class Point(BaseModel):
    timestamp: float
    value: float

class SeriesIn(BaseModel):
    series_id: str
    points: List[Tuple[float, float]]
