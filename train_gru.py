"""
Competitive GRU model for Wunder Predictorium
- Forward-only GRU (exact predictions with step-by-step caching)
- WeightedPearsonLoss (optimizes competition metric directly)
- Full training set
"""
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from collections import deque
from pathlib import Path
import time
import os

np.random.seed(42)
torch.manual_seed(42)

WORK_DIR = Path(r"D:\Wunder Fund\Claude")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)


# ============================================================================
# Feature Engineering
# ============================================================================

def engineer_features_batch(raw_features):
    """Vectorized feature engineering for full sequences.
    raw_features: (seq_len, 32)
    Returns: (seq_len, 32 + N_engineered)
    """
    p = raw_features[:, :12]   # prices
    v = raw_features[:, 12:24] # volumes
    tp = raw_features[:, 24:28] # trade prices
    tv = raw_features[:, 28:32] # trade volumes

    mid = (p[:, 0] + p[:, 6]) / 2  # best bid + best ask / 2
    spread = p[:, 6] - p[:, 0]     # ask - bid
    wmid = (p[:, 0] * v[:, 6] + p[:, 6] * v[:, 0]) / (v[:, 0] + v[:, 6] + 1e-8)
    imb_top = (v[:, 0] - v[:, 6]) / (v[:, 0] + v[:, 6] + 1e-8)
    tvol_bid = np.sum(v[:, :6], axis=1)
    tvol_ask = np.sum(v[:, 6:], axis=1)
    tvol = tvol_bid + tvol_ask
    vol_imb = (tvol_bid - tvol_ask) / (tvol + 1e-8)
    rel_spread = spread / (mid + 1e-8)

    # Multi-level imbalances
    imb_l2 = (v[:, 1] - v[:, 7]) / (v[:, 1] + v[:, 7] + 1e-8)
    imb_l3 = (v[:, 2] - v[:, 8]) / (v[:, 2] + v[:, 8] + 1e-8)

    # Price depth
    bid_depth = p[:, 0] - p[:, 5]
    ask_depth = p[:, 11] - p[:, 6]

    # Trade features
    avg_tp = np.mean(tp, axis=1)
    avg_tv = np.mean(tv, axis=1)
    trade_imb = (tp[:, 0] - tp[:, 1]) / (np.abs(tp[:, 0]) + np.abs(tp[:, 1]) + 1e-8)

    # Volume concentration
    vol_conc_bid = v[:, 0] / (tvol_bid + 1e-8)
    vol_conc_ask = v[:, 6] / (tvol_ask + 1e-8)

    # Log volumes
    log_tvol = np.log1p(np.abs(tvol))
    log_v0 = np.log1p(np.abs(v[:, 0]))

    # Rolling features (vectorized)
    N = len(raw_features)
    pmom5 = np.zeros(N)
    pvol5 = np.zeros(N)
    pmom20 = np.zeros(N)
    if N > 5:
        pmom5[5:] = mid[5:] - mid[:-5]
        # Rolling std using cumsum trick
        cumsum = np.cumsum(np.concatenate([[0], mid]))
        cumsum2 = np.cumsum(np.concatenate([[0], mid**2]))
        for w in [6]:
            s = cumsum[w:] - cumsum[:-w]
            s2 = cumsum2[w:] - cumsum2[:-w]
            pvol5[5:] = np.sqrt(np.maximum(s2/w - (s/w)**2, 0))
    if N > 20:
        pmom20[20:] = mid[20:] - mid[:-20]

    engineered = np.column_stack([
        mid, spread, wmid, imb_top,
        tvol, vol_imb, rel_spread,
        imb_l2, imb_l3,
        bid_depth, ask_depth,
        avg_tp, avg_tv, trade_imb,
        vol_conc_bid, vol_conc_ask,
        log_tvol, log_v0,
        pmom5, pvol5, pmom20
    ])

    return np.concatenate([raw_features, engineered], axis=1).astype(np.float32)


# ============================================================================
# Dataset
# ============================================================================

class LOBDataset(Dataset):
    def __init__(self, df, feat_mean=None, feat_std=None, mode='train'):
        self.sequences = df['seq_ix'].unique()
        self.mode = mode

        feat_cols = [c for c in df.columns if c not in ['seq_ix', 'step_in_seq', 'need_prediction', 't0', 't1']]
        target_cols = ['t0', 't1']

        self.data = {}
        grouped = df.sort_values(['seq_ix', 'step_in_seq']).groupby('seq_ix')
        n_done = 0
        for seq_ix, seq in grouped:
            raw = seq[feat_cols].values.astype(np.float32)
            targets = seq[target_cols].values.astype(np.float32)
            eng = engineer_features_batch(raw)
            self.data[seq_ix] = (eng, targets)
            n_done += 1
            if n_done % 1000 == 0:
                print(f"    {mode}: {n_done}/{len(self.sequences)} seqs processed", flush=True)

        # Compute normalization
        if feat_mean is None:
            all_feats = np.concatenate([d[0] for d in self.data.values()], axis=0)
            self.feat_mean = np.mean(all_feats, axis=0)
            self.feat_std = np.std(all_feats, axis=0) + 1e-8
        else:
            self.feat_mean = feat_mean
            self.feat_std = feat_std

        # Normalize
        for seq_ix in self.data:
            eng, targets = self.data[seq_ix]
            self.data[seq_ix] = ((eng - self.feat_mean) / self.feat_std, targets)

        print(f"  {mode}: {len(self.sequences)} seqs, {eng.shape[1]} features", flush=True)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_ix = self.sequences[idx]
        feats, targets = self.data[seq_ix]
        return torch.FloatTensor(feats), torch.FloatTensor(targets)


# ============================================================================
# Model
# ============================================================================

class CompetitiveGRU(nn.Module):
    """Forward-only GRU with feature extraction head"""
    def __init__(self, input_dim, hidden_dim=192, num_layers=3, dropout=0.15):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            hidden_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x):
        x = self.input_proj(x)
        gru_out, _ = self.gru(x)
        return self.head(gru_out)


# ============================================================================
# Loss
# ============================================================================

class WeightedPearsonLoss(nn.Module):
    def forward(self, pred, target):
        pred = torch.clamp(pred, -6, 6)
        total_loss = 0
        for i in range(2):
            p = pred[:, :, i].flatten()
            t = target[:, :, i].flatten()
            w = torch.abs(t) + 1e-8
            wsum = w.sum()
            pm = (p * w).sum() / wsum
            tm = (t * w).sum() / wsum
            pc = p - pm
            tc = t - tm
            num = (w * pc * tc).sum()
            den = torch.sqrt((w * pc**2).sum() * (w * tc**2).sum()) + 1e-8
            total_loss += -num / den
        return total_loss / 2


# ============================================================================
# Training
# ============================================================================

def evaluate(model, loader, device):
    model.eval()
    all_pred, all_target, all_weight = [], [], []
    with torch.no_grad():
        for feats, targets in loader:
            feats = feats.to(device)
            pred = model(feats).cpu().numpy()
            tgt = targets.numpy()
            all_pred.append(pred[:, 99:, :])
            all_target.append(tgt[:, 99:, :])
            all_weight.append(np.abs(tgt[:, 99:, :]))
    pred = np.concatenate(all_pred).reshape(-1, 2)
    tgt = np.concatenate(all_target).reshape(-1, 2)
    wgt = np.concatenate(all_weight).reshape(-1, 2)
    pred = np.clip(pred, -6, 6)

    scores = {}
    for i, name in enumerate(['t0', 't1']):
        p, t, w = pred[:, i], tgt[:, i], wgt[:, i]
        wsum = w.sum()
        pm = (p * w).sum() / wsum
        tm = (t * w).sum() / wsum
        pc, tc = p - pm, t - tm
        num = (w * pc * tc).sum()
        den = np.sqrt((w * pc**2).sum() * (w * tc**2).sum()) + 1e-8
        scores[name] = num / den
    scores['mean'] = (scores['t0'] + scores['t1']) / 2
    return scores


def train():
    N_TRAIN_SEQS = 3000  # Use subset for CPU training speed

    print("Loading data...", flush=True)
    train_df = pd.read_parquet(WORK_DIR / 'datasets' / 'train.parquet')
    valid_df = pd.read_parquet(WORK_DIR / 'datasets' / 'valid.parquet')
    print(f"  Loaded: train={len(train_df)} rows, valid={len(valid_df)} rows", flush=True)

    # Subset training data for CPU speed
    train_seqs = train_df['seq_ix'].unique()
    np.random.shuffle(train_seqs)
    train_seqs_subset = train_seqs[:N_TRAIN_SEQS]
    train_df = train_df[train_df['seq_ix'].isin(train_seqs_subset)]
    print(f"  Using {N_TRAIN_SEQS} train seqs ({len(train_df)} rows)", flush=True)

    print("Building datasets...", flush=True)
    t0 = time.time()
    train_ds = LOBDataset(train_df, mode='train')
    valid_ds = LOBDataset(valid_df, feat_mean=train_ds.feat_mean,
                          feat_std=train_ds.feat_std, mode='valid')
    print(f"  Dataset build: {time.time()-t0:.0f}s", flush=True)

    input_dim = train_ds.data[train_ds.sequences[0]][0].shape[1]
    print(f"  Input dim: {input_dim}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False, num_workers=0)

    model = CompetitiveGRU(input_dim, hidden_dim=128, num_layers=2, dropout=0.1).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {params:,}", flush=True)

    criterion = WeightedPearsonLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)

    best_score = -float('inf')
    patience = 0

    for epoch in range(15):
        model.train()
        total_loss = 0
        t0 = time.time()

        n_batches = len(train_loader)
        for batch_idx, (feats, targets) in enumerate(train_loader):
            feats, targets = feats.to(device), targets.to(device)
            optimizer.zero_grad()
            pred = model(feats)
            loss = criterion(pred[:, 99:, :], targets[:, 99:, :])
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            if (batch_idx + 1) % 20 == 0:
                print(f"  batch {batch_idx+1}/{n_batches} loss={loss.item():.4f}", flush=True)

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        scores = evaluate(model, valid_loader, device)
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:2d} | loss={avg_loss:.4f} | "
              f"score={scores['mean']:.4f} (t0={scores['t0']:.4f} t1={scores['t1']:.4f}) | "
              f"{elapsed:.0f}s", flush=True)

        if scores['mean'] > best_score:
            best_score = scores['mean']
            patience = 0
            torch.save(model.state_dict(), WORK_DIR / 'best_gru.pt')
            np.savez(WORK_DIR / 'gru_norm_stats.npz',
                     mean=train_ds.feat_mean, std=train_ds.feat_std)
            print(f"  ** BEST: {best_score:.4f} **", flush=True)
        else:
            patience += 1
            if patience >= 5:
                print("Early stopping")
                break

    print(f"\nBest validation score: {best_score:.4f}")

    # Export to ONNX
    print("Exporting ONNX...")
    model.load_state_dict(torch.load(WORK_DIR / 'best_gru.pt', map_location=device, weights_only=True))
    model.eval()
    dummy = torch.randn(1, 1000, input_dim).to(device)
    torch.onnx.export(
        model, dummy, str(WORK_DIR / 'model_gru.onnx'),
        export_params=True, opset_version=14,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamo=False
    )
    print("ONNX exported: model_gru.onnx")
    print(f"Size: {os.path.getsize(WORK_DIR / 'model_gru.onnx') / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    train()
