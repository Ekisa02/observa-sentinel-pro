import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from app.core.config import settings


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max(2, hidden_dim // 2)),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(max(2, hidden_dim // 2), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


class AnomalyDetector:
    """Wrapper for a simple autoencoder-based anomaly detector.

    This is intentionally small for demonstration; expects a 1D timeseries (values) and
    uses sliding windows to compute reconstruction error as an anomaly score.
    """
    def __init__(self, path: str = None, window: int = 16, threshold: float = 0.1):
        self.path = path or settings.MODEL_PATH
        self.window = window
        self.threshold = threshold
        self.model = None
        self.device = torch.device('cpu')

    def _make_model(self, input_dim):
        return Autoencoder(input_dim=input_dim, hidden_dim=max(8, input_dim // 2)).to(self.device)

    def load(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Model not found: {self.path}")
        saved = torch.load(self.path, map_location=self.device)
        input_dim = saved.get('input_dim')
        self.model = self._make_model(input_dim)
        self.model.load_state_dict(saved['state_dict'])
        self.model.eval()

    def save(self, path: str = None):
        path = path or self.path
        if self.model is None:
            raise RuntimeError('No model to save')
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'state_dict': self.model.state_dict(),
            'input_dim': self.model.encoder[0].in_features
        }, path)

    def train(self, sequences: List[List[float]], epochs: int = 10, lr: float = 1e-3):
        # Prepare training windows
        windows = []
        for seq in sequences:
            seq = np.array(seq, dtype=np.float32)
            if len(seq) < self.window:
                continue
            for i in range(len(seq) - self.window + 1):
                windows.append(seq[i:i+self.window])
        if not windows:
            raise ValueError('No windows available for training')
        X = torch.tensor(np.stack(windows), dtype=torch.float32).to(self.device)
        self.model = self._make_model(input_dim=self.window)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.model.train()
        for epoch in range(epochs):
            opt.zero_grad()
            out = self.model(X)
            loss = loss_fn(out, X)
            loss.backward()
            opt.step()
        self.model.eval()

    def predict(self, values: List[float]) -> Tuple[float, bool]:
        """Returns anomaly score and boolean (is_anomaly).

        We compute reconstruction error on the last window.
        """
        if self.model is None:
            raise RuntimeError('Model not loaded')
        seq = np.array(values, dtype=np.float32)
        if len(seq) < self.window:
            # not enough data; return low confidence
            return (0.0, False)
        window = seq[-self.window:]
        x = torch.tensor(window.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            out = self.model(x)
            mse = float(((out - x) ** 2).mean().item())
        is_anom = mse > self.threshold
        return (mse, is_anom)
