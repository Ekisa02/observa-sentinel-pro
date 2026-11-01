#!/usr/bin/env python3
import argparse
import json
import os
from app.services.storage import storage
from app.ml.trainer import train_from_sequences

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an Observa Sentinel model from stored series')
    parser.add_argument('--output', '-o', default='models/autoencoder.pt')
    parser.add_argument('--epochs', '-e', default=5, type=int)
    args = parser.parse_args()

    seqs = storage.all_series_values()
    if not seqs:
        print('No data stored. For demo, generating synthetic data...')
        # generate simple sin + noise sequences
        import numpy as np
        seqs = []
        for i in range(10):
            t = np.linspace(0, 10, 200)
            vals = np.sin(t * (1 + i*0.1)) + np.random.normal(scale=0.05, size=t.shape)
            seqs.append(vals.tolist())

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    path = train_from_sequences(seqs, args.output, epochs=args.epochs)
    print(f'Model saved to {path}')
