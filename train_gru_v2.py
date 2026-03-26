"""
GRU v2: Enhanced features (200+) instead of 53.
Key insight: GRU alone scored 0.2662 test, ensemble with LGB was 0.2657 (worse!).
So improving the GRU itself is the path forward.

Changes from v1:
  - ~200+ features (same as tree models): rolling stats, lags, diffs, EWM
  - More training sequences (5000)
  - Larger hidden dim (192), 3 layers
  - 20 epochs
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

np.random.seed(42)
torch.manual_seed(42)

WORK_DIR = Path(r"D:\Wunder Fund\Claude")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

N_TRAIN_SEQS = 3000
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.1
LR = 2e-3
EPOCHS = 15
BATCH_SIZE = 32
PATIENCE = 5


# ============================================================================
# Rich Feature Engineering (200+ features, same as tree models)
# ============================================================================

def engineer_features_rich(raw_features):
    """Vectorized feature engineering for full sequences.
    raw_features: (seq_len, 32) -> (seq_len, ~210)
    Uses the same features as tree models for maximum info.
    """
    N = len(raw_features)
    p = raw_features[:, :12]
    v = raw_features[:, 12:24]
    tp = raw_features[:, 24:28]
    tv = raw_features[:, 28:32]

    # ---- Base features ----
    mid = (p[:, 0] + p[:, 6]) / 2
    spread = p[:, 6] - p[:, 0]
    rel_spread = spread / (np.abs(mid) + 1e-8)
    wmid = (p[:, 0] * v[:, 6] + p[:, 6] * v[:, 0]) / (v[:, 0] + v[:, 6] + 1e-8)
    wmid_diff = wmid - mid

    # Imbalances at each level
    imb = [(v[:, i] - v[:, 6+i]) / (v[:, i] + v[:, 6+i] + 1e-8) for i in range(6)]

    # Volume aggregates
    tvol_bid = np.sum(v[:, :6], axis=1)
    tvol_ask = np.sum(v[:, 6:], axis=1)
    tvol = tvol_bid + tvol_ask
    vol_imb = (tvol_bid - tvol_ask) / (tvol + 1e-8)
    log_tvol = np.log1p(np.abs(tvol))
    vol_conc_bid = v[:, 0] / (tvol_bid + 1e-8)
    vol_conc_ask = v[:, 6] / (tvol_ask + 1e-8)
    vol_conc_bid2 = (v[:, 0] + v[:, 1]) / (tvol_bid + 1e-8)
    vol_conc_ask2 = (v[:, 6] + v[:, 7]) / (tvol_ask + 1e-8)

    # Depth
    bid_depth = p[:, 0] - p[:, 5]
    ask_depth = p[:, 11] - p[:, 6]
    depth_imb = bid_depth - ask_depth

    # Gaps
    gaps = []
    for i in range(5):
        gaps.append(p[:, i] - p[:, i+1])
        gaps.append(p[:, 6+i+1] - p[:, 6+i])

    # Trades
    avg_tp = np.mean(tp, axis=1)
    avg_tv = np.mean(tv, axis=1)
    trade_imb_feat = (tp[:, 0] - tp[:, 1]) / (np.abs(tp[:, 0]) + np.abs(tp[:, 1]) + 1e-8)
    trade_intensity = np.sum(np.abs(tv), axis=1)
    signed_flow = np.sum(tp * tv, axis=1)
    tp_vs_mid = avg_tp - mid

    # VWAP
    vwap_bid = np.sum(p[:, :6] * v[:, :6], axis=1) / (tvol_bid + 1e-8)
    vwap_ask = np.sum(p[:, 6:] * v[:, 6:], axis=1) / (tvol_ask + 1e-8)
    vwap_spread = vwap_ask - vwap_bid
    log_v0 = np.log1p(np.abs(v[:, 0]))
    log_v6 = np.log1p(np.abs(v[:, 6]))

    # Pressure
    press_bid = v[:, 0] * (p[:, 0] - p[:, 1])
    press_ask = v[:, 6] * (p[:, 7] - p[:, 6])
    press_imb = press_bid - press_ask

    # Interactions
    spread_x_vol_imb = spread * vol_imb
    imb_l0_x_tvol = imb[0] * tvol
    press_imb_x_vol_imb = press_imb * vol_imb
    spread_x_imb_l0 = spread * imb[0]
    trade_flow_x_imb = signed_flow * imb[0]
    spread_to_depth = spread / (bid_depth + ask_depth + 1e-8)
    vol_ratio = tvol_bid / (tvol_ask + 1e-8)
    vwap_mid_diff = (vwap_bid + vwap_ask) / 2 - mid

    step_norm = np.arange(N, dtype=np.float32) / 999.0

    # Collect base features
    base_list = [
        mid, spread, rel_spread, wmid, wmid_diff,
        *imb,  # 6 features
        tvol_bid, tvol_ask, tvol, vol_imb, log_tvol,
        vol_conc_bid, vol_conc_ask, vol_conc_bid2, vol_conc_ask2,
        bid_depth, ask_depth, depth_imb,
        *gaps,  # 10 features
        avg_tp, avg_tv, trade_imb_feat, trade_intensity, signed_flow, tp_vs_mid,
        vwap_bid, vwap_ask, vwap_spread, log_v0, log_v6,
        press_bid, press_ask, press_imb,
        spread_x_vol_imb, imb_l0_x_tvol, press_imb_x_vol_imb,
        spread_x_imb_l0, trade_flow_x_imb,
        spread_to_depth, vol_ratio, vwap_mid_diff,
        step_norm,
    ]

    # ---- Rolling features ----
    def rolling_mean_1d(arr, w):
        cs = np.cumsum(np.concatenate([[0], arr]))
        out = np.zeros(N, dtype=np.float32)
        out[w-1:] = (cs[w:] - cs[:-w]) / w
        for i in range(min(w-1, N)):
            out[i] = cs[i+1] / (i + 1)
        return out

    def rolling_std_1d(arr, w):
        m = rolling_mean_1d(arr, w)
        m2 = rolling_mean_1d(arr**2, w)
        return np.sqrt(np.maximum(m2 - m**2, 0))

    key_series = {'mid': mid, 'spread': spread, 'vol_imb': vol_imb,
                  'tvol': tvol, 'imb_l0': imb[0]}
    trade_series = {'signed_flow': signed_flow, 'trade_intensity': trade_intensity,
                    'press_imb': press_imb}

    rolling_feats = []
    for w in [5, 10, 20, 50, 100]:
        for name, arr in key_series.items():
            rolling_feats.append(rolling_mean_1d(arr, w))
            rolling_feats.append(rolling_std_1d(arr, w))
        # Mid momentum
        mom = np.zeros(N, dtype=np.float32)
        if w < N:
            mom[w:] = mid[w:] - mid[:-w]
        rolling_feats.append(mom)

    # Trade rolling means
    for w in [5, 10, 20]:
        for name, arr in trade_series.items():
            rolling_feats.append(rolling_mean_1d(arr, w))

    # ---- Lag/diff features ----
    lag_diff_feats = []
    for name, arr in key_series.items():
        for k in [1, 2, 3, 5]:
            lagged = np.zeros(N, dtype=np.float32)
            if k < N:
                lagged[k:] = arr[:-k]
            lag_diff_feats.append(lagged)
            lag_diff_feats.append(arr - lagged)  # diff

    # Pairwise lag diffs
    for name in ['mid', 'vol_imb', 'imb_l0']:
        arr = key_series[name]
        lags = {}
        for k in [1, 2, 3, 5]:
            l = np.zeros(N, dtype=np.float32)
            if k < N:
                l[k:] = arr[:-k]
            lags[k] = l
        lag_diff_feats.extend([
            lags[1] - lags[2], lags[1] - lags[3], lags[2] - lags[3],
            lags[1] - lags[5], lags[2] - lags[5],
        ])

    # ---- EWM features ----
    ewm_feats = []
    for name in ['mid', 'vol_imb', 'imb_l0']:
        arr = key_series[name]
        for span in [5, 20]:
            alpha = 2.0 / (span + 1.0)
            ewm = np.zeros(N, dtype=np.float32)
            ewm[0] = arr[0]
            for t in range(1, N):
                ewm[t] = alpha * arr[t] + (1 - alpha) * ewm[t-1]
            ewm_feats.append(ewm)
            ewm_feats.append(arr - ewm)  # deviation

    # Combine all
    all_extra = base_list + rolling_feats + lag_diff_feats + ewm_feats
    extra_mat = np.column_stack(all_extra)

    result = np.concatenate([raw_features, extra_mat], axis=1).astype(np.float32)
    return result


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
            eng = engineer_features_rich(raw)
            self.data[seq_ix] = (eng, targets)
            n_done += 1
            if n_done % 500 == 0:
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

        dim = self.data[self.sequences[0]][0].shape[1]
        print(f"  {mode}: {len(self.sequences)} seqs, {dim} features", flush=True)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_ix = self.sequences[idx]
        feats, targets = self.data[seq_ix]
        return torch.FloatTensor(feats), torch.FloatTensor(targets)


# ============================================================================
# Model (same architecture, but bigger to handle more features)
# ============================================================================

class CompetitiveGRU(nn.Module):
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
# Loss & Evaluation
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
    print(f"GRU v2: Enhanced features, {N_TRAIN_SEQS} seqs, hidden={HIDDEN_DIM}", flush=True)

    print("\nLoading data...", flush=True)
    train_df = pd.read_parquet(WORK_DIR / 'datasets' / 'train.parquet')
    valid_df = pd.read_parquet(WORK_DIR / 'datasets' / 'valid.parquet')
    print(f"  train={len(train_df)} rows, valid={len(valid_df)} rows", flush=True)

    train_seqs = train_df['seq_ix'].unique()
    np.random.shuffle(train_seqs)
    train_df = train_df[train_df['seq_ix'].isin(train_seqs[:N_TRAIN_SEQS])]
    print(f"  Using {N_TRAIN_SEQS} train seqs ({len(train_df)} rows)", flush=True)

    print("\nBuilding datasets...", flush=True)
    t0 = time.time()
    train_ds = LOBDataset(train_df, mode='train')
    del train_df
    valid_ds = LOBDataset(valid_df, feat_mean=train_ds.feat_mean,
                          feat_std=train_ds.feat_std, mode='valid')
    del valid_df
    print(f"  Dataset build: {time.time()-t0:.0f}s", flush=True)

    input_dim = train_ds.data[train_ds.sequences[0]][0].shape[1]
    print(f"  Input dim: {input_dim} (vs 53 in v1)", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CompetitiveGRU(input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                           dropout=DROPOUT).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {params:,}", flush=True)

    criterion = WeightedPearsonLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_score = -float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        t_start = time.time()
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
            if (batch_idx + 1) % 50 == 0:
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
            torch.save(model.state_dict(), WORK_DIR / 'best_gru_v2.pt')
            np.savez(WORK_DIR / 'gru_v2_norm_stats.npz',
                     mean=train_ds.feat_mean, std=train_ds.feat_std)
            print(f"  ** BEST: {best_score:.4f} **", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping", flush=True)
                break

    print(f"\nBest validation score: {best_score:.4f} (vs v1: ~0.246)", flush=True)

    # Export incremental ONNX (for step-by-step inference)
    print("\nExporting incremental ONNX...", flush=True)
    model.load_state_dict(torch.load(WORK_DIR / 'best_gru_v2.pt', map_location=device, weights_only=True))
    model.eval()

    # Incremental model wrapper
    class IncrGRU(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.input_proj = base.input_proj
            self.gru = base.gru
            self.head = base.head

        def forward(self, x_step, hidden_in):
            x = self.input_proj(x_step)
            gru_out, hidden_out = self.gru(x, hidden_in)
            pred = self.head(gru_out)
            return pred, hidden_out

    incr_model = IncrGRU(model).to(device)
    incr_model.eval()

    x_step = torch.randn(1, 1, input_dim).to(device)
    h_in = torch.zeros(NUM_LAYERS, 1, HIDDEN_DIM).to(device)

    onnx_path = str(WORK_DIR / 'model_gru_v2_incr.onnx')
    torch.onnx.export(
        incr_model, (x_step, h_in), onnx_path,
        export_params=True, opset_version=14,
        do_constant_folding=True,
        input_names=['x_step', 'hidden_in'],
        output_names=['pred', 'hidden_out'],
        dynamic_axes={
            'x_step': {0: 'batch'},
            'hidden_in': {1: 'batch'},
            'pred': {0: 'batch'},
            'hidden_out': {1: 'batch'},
        },
    )
    onnx_size = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"  ONNX saved: {onnx_path} ({onnx_size:.2f} MB)", flush=True)

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    with torch.no_grad():
        pt_pred, pt_h = incr_model(x_step, h_in)
    onnx_pred, onnx_h = sess.run(None, {
        'x_step': x_step.cpu().numpy(),
        'hidden_in': h_in.cpu().numpy()
    })
    max_diff = max(np.max(np.abs(pt_pred.cpu().numpy() - onnx_pred)),
                   np.max(np.abs(pt_h.cpu().numpy() - onnx_h)))
    print(f"  ONNX max diff: {max_diff:.2e}", flush=True)

    print(f"\nDone! Feature dim: {input_dim}, Model: GRU({HIDDEN_DIM})x{NUM_LAYERS}", flush=True)
    print(f"Files: best_gru_v2.pt, gru_v2_norm_stats.npz, model_gru_v2_incr.onnx", flush=True)


if __name__ == '__main__':
    train()
