"""
Dual BiGRU + CNN1d for LOB prediction.

Architecture (from user's diagram "Финальная версия"):
  - Price branch: raw_prices(12) + trade_prices(4) + shared_engineered(21) → proj → BiGRU
  - Volume branch: raw_volumes(12) + trade_volumes(4) + shared_engineered(21) → proj → BiGRU
  - Shared engineered features (21) go to BOTH branches ("Общие фичи")
  - Concat BiGRU outputs → CNN1d (k=3,5,7 progressive) → MLP head → 2 targets

Inference: batch mode (buffer all steps, periodic reprocessing via ONNX)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
import time
import os
import gc

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.benchmark = True

WORK_DIR = Path(r"D:\Wunder Fund\Claude")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = device.type == 'cuda'
print(f"Device: {device}, AMP: {USE_AMP}", flush=True)


# ============================================================================
# Feature Engineering (53 features: 32 raw + 21 engineered)
# ============================================================================

def engineer_features_batch(raw_features):
    """Vectorized feature engineering. raw_features: (seq_len, 32) -> (seq_len, 53)"""
    p = raw_features[:, :12]
    v = raw_features[:, 12:24]
    tp = raw_features[:, 24:28]
    tv = raw_features[:, 28:32]

    mid = (p[:, 0] + p[:, 6]) / 2
    spread = p[:, 6] - p[:, 0]
    wmid = (p[:, 0] * v[:, 6] + p[:, 6] * v[:, 0]) / (v[:, 0] + v[:, 6] + 1e-8)
    imb_top = (v[:, 0] - v[:, 6]) / (v[:, 0] + v[:, 6] + 1e-8)
    tvol_bid = np.sum(v[:, :6], axis=1)
    tvol_ask = np.sum(v[:, 6:], axis=1)
    tvol = tvol_bid + tvol_ask
    vol_imb = (tvol_bid - tvol_ask) / (tvol + 1e-8)
    rel_spread = spread / (mid + 1e-8)

    imb_l2 = (v[:, 1] - v[:, 7]) / (v[:, 1] + v[:, 7] + 1e-8)
    imb_l3 = (v[:, 2] - v[:, 8]) / (v[:, 2] + v[:, 8] + 1e-8)
    bid_depth = p[:, 0] - p[:, 5]
    ask_depth = p[:, 11] - p[:, 6]
    avg_tp = np.mean(tp, axis=1)
    avg_tv = np.mean(tv, axis=1)
    trade_imb = (tp[:, 0] - tp[:, 1]) / (np.abs(tp[:, 0]) + np.abs(tp[:, 1]) + 1e-8)
    vol_conc_bid = v[:, 0] / (tvol_bid + 1e-8)
    vol_conc_ask = v[:, 6] / (tvol_ask + 1e-8)
    log_tvol = np.log1p(np.abs(tvol))
    log_v0 = np.log1p(np.abs(v[:, 0]))

    N = len(raw_features)
    pmom5 = np.zeros(N)
    pvol5 = np.zeros(N)
    pmom20 = np.zeros(N)
    if N > 5:
        pmom5[5:] = mid[5:] - mid[:-5]
        cumsum = np.cumsum(np.concatenate([[0], mid]))
        cumsum2 = np.cumsum(np.concatenate([[0], mid**2]))
        s = cumsum[6:] - cumsum[:-6]
        s2 = cumsum2[6:] - cumsum2[:-6]
        pvol5[5:] = np.sqrt(np.maximum(s2/6 - (s/6)**2, 0))
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

        self.data = {}
        grouped = df.sort_values(['seq_ix', 'step_in_seq']).groupby('seq_ix')
        n_done = 0
        for seq_ix, seq in grouped:
            raw = seq[feat_cols].values.astype(np.float32)
            targets = seq[['t0', 't1']].values.astype(np.float32)
            eng = engineer_features_batch(raw)
            self.data[seq_ix] = (eng, targets)
            n_done += 1
            if n_done % 1000 == 0:
                print(f"    {mode}: {n_done}/{len(self.sequences)} seqs", flush=True)

        if feat_mean is None:
            all_feats = np.concatenate([d[0] for d in self.data.values()], axis=0)
            self.feat_mean = np.mean(all_feats, axis=0)
            self.feat_std = np.std(all_feats, axis=0) + 1e-8
        else:
            self.feat_mean = feat_mean
            self.feat_std = feat_std

        for seq_ix in self.data:
            eng, targets = self.data[seq_ix]
            self.data[seq_ix] = ((eng - self.feat_mean) / self.feat_std, targets)

        print(f"  {mode}: {len(self.sequences)} seqs, features={self.data[self.sequences[0]][0].shape[1]}", flush=True)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_ix = self.sequences[idx]
        feats, targets = self.data[seq_ix]
        return torch.FloatTensor(feats), torch.FloatTensor(targets)


# ============================================================================
# Model: Dual BiGRU + CNN1d (user's "Final Version" architecture)
# ============================================================================

class DualBiGRU_CNN(nn.Module):
    """
    Two-branch architecture:
      Branch 1 (Price): prices(12) + trade_prices(4) + shared_eng(21) = 37 dims → BiGRU
      Branch 2 (Volume): volumes(12) + trade_volumes(4) + shared_eng(21) = 37 dims → BiGRU
    Fusion:
      Concat(price_gru_out, volume_gru_out) → CNN1d(k=3,5,7) → MLP head → 2 targets
    """
    def __init__(self, hidden_dim=96, num_layers=2, dropout=0.15):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ---- Price branch (37 features) ----
        self.price_proj = nn.Sequential(
            nn.Linear(37, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.price_gru = nn.GRU(
            hidden_dim, hidden_dim // 2,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.price_norm = nn.LayerNorm(hidden_dim)

        # ---- Volume branch (37 features) ----
        self.volume_proj = nn.Sequential(
            nn.Linear(37, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.volume_gru = nn.GRU(
            hidden_dim, hidden_dim // 2,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.volume_norm = nn.LayerNorm(hidden_dim)

        # ---- CNN1d fusion (progressive bottleneck) ----
        cat_dim = hidden_dim * 2  # concat of both GRU outputs
        self.cnn = nn.Sequential(
            nn.Conv1d(cat_dim, cat_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(cat_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=7, padding=3),
            nn.GELU(),
        )
        self.cnn_norm = nn.LayerNorm(hidden_dim // 2)

        # ---- Prediction head ----
        self.head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 2),
        )

    def forward(self, x):
        # x: (batch, seq_len, 53)
        # Split into domain-specific + shared engineered features
        price_raw = x[:, :, :12]       # 12 price levels
        vol_raw = x[:, :, 12:24]       # 12 volume levels
        price_trade = x[:, :, 24:28]   # 4 trade prices
        vol_trade = x[:, :, 28:32]     # 4 trade volumes
        shared_eng = x[:, :, 32:]      # 21 shared engineered features

        # Price branch: prices + trade_prices + shared
        x_price = torch.cat([price_raw, price_trade, shared_eng], dim=-1)  # 37
        p = self.price_proj(x_price)
        p, _ = self.price_gru(p)
        p = self.price_norm(p)  # (B, T, hidden_dim)

        # Volume branch: volumes + trade_volumes + shared
        x_vol = torch.cat([vol_raw, vol_trade, shared_eng], dim=-1)  # 37
        v = self.volume_proj(x_vol)
        v, _ = self.volume_gru(v)
        v = self.volume_norm(v)  # (B, T, hidden_dim)

        # Concat + CNN1d
        cat = torch.cat([p, v], dim=-1)        # (B, T, 2*hidden)
        cnn_in = cat.transpose(1, 2)           # (B, 2*hidden, T)
        cnn_out = self.cnn(cnn_in)             # (B, hidden//2, T)
        cnn_out = cnn_out.transpose(1, 2)      # (B, T, hidden//2)
        cnn_out = self.cnn_norm(cnn_out)

        # Head
        output = self.head(cnn_out)  # (B, T, 2)
        return output


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
# Evaluation
# ============================================================================

def evaluate(model, loader, device):
    model.eval()
    all_pred, all_target = [], []
    with torch.no_grad():
        for feats, targets in loader:
            feats = feats.to(device)
            pred = model(feats).cpu().numpy()
            tgt = targets.numpy()
            all_pred.append(pred[:, 99:, :])
            all_target.append(tgt[:, 99:, :])
            del feats
    gc.collect()
    pred = np.concatenate(all_pred).reshape(-1, 2)
    tgt = np.concatenate(all_target).reshape(-1, 2)
    pred = np.clip(pred, -6, 6)

    scores = {}
    for i, name in enumerate(['t0', 't1']):
        p, t = pred[:, i], tgt[:, i]
        w = np.abs(t) + 1e-8
        wsum = w.sum()
        pm = (p * w).sum() / wsum
        tm = (t * w).sum() / wsum
        pc, tc = p - pm, t - tm
        num = (w * pc * tc).sum()
        den = np.sqrt((w * pc**2).sum() * (w * tc**2).sum()) + 1e-8
        scores[name] = num / den
    scores['mean'] = (scores['t0'] + scores['t1']) / 2
    return scores


# ============================================================================
# Training
# ============================================================================

def train():
    if device.type == 'cuda':
        N_TRAIN_SEQS = 4000
        HIDDEN_DIM = 96
        BATCH_SIZE = 64
        NUM_LAYERS = 2
        EPOCHS = 15
        PATIENCE = 6
    else:
        # CPU-optimized: smaller model, fewer seqs for feasible training
        N_TRAIN_SEQS = 1500
        HIDDEN_DIM = 64
        BATCH_SIZE = 16
        NUM_LAYERS = 1
        EPOCHS = 8
        PATIENCE = 4
    LR = 2e-3

    print("Loading data...", flush=True)
    train_df = pd.read_parquet(WORK_DIR / 'datasets' / 'train.parquet')
    valid_df = pd.read_parquet(WORK_DIR / 'datasets' / 'valid.parquet')
    print(f"  train={len(train_df)} rows, valid={len(valid_df)} rows", flush=True)

    train_seqs = train_df['seq_ix'].unique()
    np.random.shuffle(train_seqs)
    train_df = train_df[train_df['seq_ix'].isin(train_seqs[:N_TRAIN_SEQS])]
    print(f"  Using {N_TRAIN_SEQS} train seqs ({len(train_df)} rows)", flush=True)

    # Also limit validation to save memory on CPU
    if device.type != 'cuda':
        valid_seqs = valid_df['seq_ix'].unique()
        np.random.shuffle(valid_seqs)
        valid_df = valid_df[valid_df['seq_ix'].isin(valid_seqs[:500])]
        print(f"  Using 500 valid seqs ({len(valid_df)} rows)", flush=True)

    print("Building datasets...", flush=True)
    t0 = time.time()
    train_ds = LOBDataset(train_df, mode='train')
    del train_df; gc.collect()
    valid_ds = LOBDataset(valid_df, feat_mean=train_ds.feat_mean,
                          feat_std=train_ds.feat_std, mode='valid')
    del valid_df; gc.collect()
    print(f"  Dataset build: {time.time()-t0:.0f}s", flush=True)

    input_dim = train_ds.data[train_ds.sequences[0]][0].shape[1]
    print(f"  Input dim: {input_dim} (expecting 53)", flush=True)
    assert input_dim == 53, f"Expected 53 features, got {input_dim}"

    pin = device.type == 'cuda'
    EVAL_BATCH = BATCH_SIZE if device.type == 'cuda' else 8
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=pin)
    valid_loader = DataLoader(valid_ds, batch_size=EVAL_BATCH, shuffle=False,
                              num_workers=0, pin_memory=pin)

    # Model
    model = DualBiGRU_CNN(
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=0.15
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {params:,}", flush=True)
    print(f"  Architecture: Dual BiGRU(price+volume) + CNN1d(k=3,5,7) + Head", flush=True)

    criterion = WeightedPearsonLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    best_score = -float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        t_start = time.time()
        n_batches = len(train_loader)

        for batch_idx, (feats, targets) in enumerate(train_loader):
            feats, targets = feats.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                pred = model(feats)
                loss = criterion(pred[:, 99:, :], targets[:, 99:, :])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            if (batch_idx + 1) % 20 == 0:
                print(f"  batch {batch_idx+1}/{n_batches} loss={loss.item():.4f}", flush=True)

        scheduler.step()
        avg_loss = total_loss / n_batches

        scores = evaluate(model, valid_loader, device)
        elapsed = time.time() - t_start

        print(f"Epoch {epoch+1:2d} | loss={avg_loss:.4f} | "
              f"score={scores['mean']:.4f} (t0={scores['t0']:.4f} t1={scores['t1']:.4f}) | "
              f"{elapsed:.0f}s", flush=True)

        if scores['mean'] > best_score:
            best_score = scores['mean']
            patience_counter = 0
            torch.save(model.state_dict(), WORK_DIR / 'best_dual_bigru_cnn.pt')
            np.savez(WORK_DIR / 'dual_bigru_cnn_norm_stats.npz',
                     mean=train_ds.feat_mean, std=train_ds.feat_std)
            print(f"  ** BEST: {best_score:.4f} **", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping", flush=True)
                break

    print(f"\nBest validation score: {best_score:.4f}", flush=True)

    # ========================================================================
    # Export ONNX (batch mode, full sequences)
    # ========================================================================
    print("\nExporting ONNX (batch mode)...", flush=True)
    model.load_state_dict(torch.load(WORK_DIR / 'best_dual_bigru_cnn.pt',
                                     map_location=device, weights_only=True))
    model.eval()

    dummy = torch.randn(1, 1000, 53).to(device)
    onnx_path = str(WORK_DIR / 'model_dual_bigru_cnn.onnx')
    try:
        torch.onnx.export(
            model, dummy, onnx_path,
            export_params=True, opset_version=14,
            do_constant_folding=True,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch', 1: 'seq_len'},
                          'output': {0: 'batch', 1: 'seq_len'}},
            dynamo=False,
        )
    except TypeError:
        # older PyTorch without dynamo param
        torch.onnx.export(
            model, dummy, onnx_path,
            export_params=True, opset_version=14,
            do_constant_folding=True,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch', 1: 'seq_len'},
                          'output': {0: 'batch', 1: 'seq_len'}},
        )
    onnx_size = os.path.getsize(onnx_path) / 1024
    print(f"  ONNX saved: {onnx_path} ({onnx_size:.1f} KB)", flush=True)

    # Verify ONNX
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    with torch.no_grad():
        pt_out = model(dummy).cpu().numpy()
    onnx_out = sess.run(None, {'input': dummy.cpu().numpy()})[0]
    max_diff = np.max(np.abs(pt_out - onnx_out))
    print(f"  ONNX vs PyTorch max diff: {max_diff:.2e}", flush=True)

    print(f"\nDone! Best score: {best_score:.4f}", flush=True)
    print(f"Compare: GRU v1 validation ~0.246, GRU v2 validation ~0.248", flush=True)


if __name__ == '__main__':
    train()
