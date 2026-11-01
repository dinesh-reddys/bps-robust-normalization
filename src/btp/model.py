"""MLP architecture adapted from the BPS authors' classification demo.

https://github.com/sergeyprokudin/bps/blob/master/bps_demos/train_modelnet_mlp.py
The author demo is MIT-0 licensed. Attribution is recorded in SOURCES.md.
"""
import torch
from torch import nn


class BPSMLP(nn.Module):
    def __init__(self, n_features=512, n_classes=40, hidden_sizes=(512, 512), dropout=0.8):
        super().__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, hidden_sizes[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[1], n_classes),
        )

    def forward(self, x):
        return self.network(x)


def seed_everything(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
