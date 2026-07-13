"""
Bidirectional LSTM trained with leave-one-patient-out (LOPO) cross-validation.

For each held-out patient, the model is trained on all other patients (optionally with
synthetic augmentation of the training fold only) and scored on the held-out patient.
Predicted probabilities for all held-out windows are concatenated for evaluation.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 32), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def train_one_fold(Xtr, ytr, Xval, *, epochs=30, lr=1e-3, device="cpu"):
    model = BiLSTM(Xtr.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # class imbalance handled via pos_weight
    pos = max(1, int(ytr.sum())); neg = max(1, len(ytr) - pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, dtype=torch.float32, device=device))
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xtr_t), ytr_t)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(Xval, dtype=torch.float32, device=device)))
    return prob.cpu().numpy()


def lopo_predict(X, y, pid, augment_fn=None, **kw):
    """Leave-one-patient-out predictions. augment_fn(Xtr, ytr) -> (Xtr', ytr')."""
    prob = np.zeros(len(y), dtype="float32")
    for p in np.unique(pid):
        te = pid == p; tr = ~te
        Xtr, ytr = X[tr], y[tr]
        if augment_fn is not None:
            Xtr, ytr = augment_fn(Xtr, ytr)
        prob[te] = train_one_fold(Xtr, ytr, X[te], **kw)
    return prob
