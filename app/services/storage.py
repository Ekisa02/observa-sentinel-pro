import threading
from collections import deque, defaultdict
from typing import List, Tuple
import csv

class Storage:
    """Lightweight in-memory storage with optional disk flush for demonstration."""
    def __init__(self, limit: int = 10000):
        self.lock = threading.RLock()
        self.limit = limit
        self._store = defaultdict(deque)  # series_id -> deque of points

    def add_series(self, series_id: str, points: List[Tuple[float, float]]):
        with self.lock:
            d = self._store[series_id]
            for p in points:
                d.append(p)
            # enforce limit
            while len(d) > self.limit:
                d.popleft()

    def get_series(self, series_id: str):
        with self.lock:
            return list(self._store.get(series_id, []))

    def all_series_values(self):
        """Return list of numeric lists for training: only values."""
        with self.lock:
            return [[float(pt[1]) for pt in pts] for pts in self._store.values() if pts]

    def flush_to_csv(self, path: str):
        with self.lock:
            with open(path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["series_id", "timestamp", "value"])
                for sid, pts in self._store.items():
                    for t, v in pts:
                        writer.writerow([sid, t, v])

storage = Storage()
