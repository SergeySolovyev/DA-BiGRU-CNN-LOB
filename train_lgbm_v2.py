"""LightGBM v2: train on ALL sequences with row subsampling.
Key changes from v1:
  - Use ALL 10721 train sequences (was 2000)
  - 30% row subsample to manage memory
  - Larger early_stop (100) for t1
  - Save models as lgbm_v2_t0.txt / lgbm_v2_t1.txt
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
import os
import json
import gc
from pathlib import Path

WORK_DIR = Path(r"D:\Wunder Fund\Claude")
TRAIN_PATH = WORK_DIR / "datasets" / "train.parquet"
VALID_PATH = WORK_DIR / "datasets" / "valid.parquet"

SEQ_LEN = 1000
N_TRAIN_SEQS = None  # None = use all


def rolling_mean(arr2d, w):
    cs = np.cumsum(arr2d, axis=1)
    out = np.zeros_like(arr2d)
    out[:, w-1:] = (cs[:, w-1:] - np.concatenate([np.zeros((arr2d.shape[0], 1)), cs[:, :-w]], axis=1)[:, :arr2d.shape[1]-w+1]) / w
    for i in range(min(w-1, arr2d.shape[1])):
        out[:, i] = cs[:, i] / (i + 1)
    return out

def rolling_std(arr2d, w):
    mean = rolling_mean(arr2d, w)
    mean_sq = rolling_mean(arr2d ** 2, w)
    var = np.maximum(mean_sq - mean ** 2, 0)
    return np.sqrt(var)

def rolling_diff(arr2d, w):
    out = np.zeros_like(arr2d)
    if w < arr2d.shape[1]:
        out[:, w:] = arr2d[:, w:] - arr2d[:, :-w]
    return out

def lag(arr2d, k=1):
    out = np.zeros_like(arr2d)
    if k < arr2d.shape[1]:
        out[:, k:] = arr2d[:, :-k]
    return out

def ewm_mean(arr2d, span):
    alpha = 2.0 / (span + 1.0)
    out = np.zeros_like(arr2d)
    out[:, 0] = arr2d[:, 0]
    for t in range(1, arr2d.shape[1]):
        out[:, t] = alpha * arr2d[:, t] + (1 - alpha) * out[:, t-1]
    return out


def engineer_features_fast(df, n_seqs):
    print("  Engineering features (vectorized)...", flush=True)
    t0_time = time.time()
    N = len(df)
    assert N == n_seqs * SEQ_LEN

    def col2d(name):
        return df[name].values.reshape(n_seqs, SEQ_LEN)

    p = np.stack([col2d(f'p{i}') for i in range(12)], axis=-1)
    v = np.stack([col2d(f'v{i}') for i in range(12)], axis=-1)
    tp = np.stack([col2d(f'dp{i}') for i in range(4)], axis=-1)
    tv = np.stack([col2d(f'dv{i}') for i in range(4)], axis=-1)

    feats = {}
    mid = (p[:,:,0] + p[:,:,6]) / 2
    spread = p[:,:,6] - p[:,:,0]
    feats['mid'] = mid
    feats['spread'] = spread
    feats['rel_spread'] = spread / (np.abs(mid) + 1e-8)
    wmid = (p[:,:,0] * v[:,:,6] + p[:,:,6] * v[:,:,0]) / (v[:,:,0] + v[:,:,6] + 1e-8)
    feats['wmid'] = wmid
    feats['wmid_diff'] = wmid - mid

    for i in range(6):
        feats[f'imb_l{i}'] = (v[:,:,i] - v[:,:,6+i]) / (v[:,:,i] + v[:,:,6+i] + 1e-8)

    tvol_bid = np.sum(v[:,:,:6], axis=-1)
    tvol_ask = np.sum(v[:,:,6:], axis=-1)
    tvol = tvol_bid + tvol_ask
    vol_imb = (tvol_bid - tvol_ask) / (tvol + 1e-8)
    feats['tvol_bid'] = tvol_bid
    feats['tvol_ask'] = tvol_ask
    feats['tvol'] = tvol
    feats['vol_imb'] = vol_imb
    feats['log_tvol'] = np.log1p(np.abs(tvol))
    feats['vol_conc_bid'] = v[:,:,0] / (tvol_bid + 1e-8)
    feats['vol_conc_ask'] = v[:,:,6] / (tvol_ask + 1e-8)
    feats['vol_conc_bid2'] = (v[:,:,0] + v[:,:,1]) / (tvol_bid + 1e-8)
    feats['vol_conc_ask2'] = (v[:,:,6] + v[:,:,7]) / (tvol_ask + 1e-8)

    feats['bid_depth'] = p[:,:,0] - p[:,:,5]
    feats['ask_depth'] = p[:,:,11] - p[:,:,6]
    feats['depth_imb'] = feats['bid_depth'] - feats['ask_depth']
    for i in range(5):
        feats[f'bid_gap_{i}'] = p[:,:,i] - p[:,:,i+1]
        feats[f'ask_gap_{i}'] = p[:,:,6+i+1] - p[:,:,6+i]

    feats['avg_tp'] = np.mean(tp, axis=-1)
    feats['avg_tv'] = np.mean(tv, axis=-1)
    feats['trade_imb'] = (tp[:,:,0] - tp[:,:,1]) / (np.abs(tp[:,:,0]) + np.abs(tp[:,:,1]) + 1e-8)
    feats['trade_intensity'] = np.sum(np.abs(tv), axis=-1)
    feats['signed_flow'] = np.sum(tp * tv, axis=-1)
    feats['tp_vs_mid'] = feats['avg_tp'] - mid

    vwap_bid = np.sum(p[:,:,:6] * v[:,:,:6], axis=-1) / (tvol_bid + 1e-8)
    vwap_ask = np.sum(p[:,:,6:] * v[:,:,6:], axis=-1) / (tvol_ask + 1e-8)
    feats['vwap_bid'] = vwap_bid
    feats['vwap_ask'] = vwap_ask
    feats['vwap_spread'] = vwap_ask - vwap_bid
    feats['log_v0'] = np.log1p(np.abs(v[:,:,0]))
    feats['log_v6'] = np.log1p(np.abs(v[:,:,6]))

    feats['press_bid'] = v[:,:,0] * (p[:,:,0] - p[:,:,1])
    feats['press_ask'] = v[:,:,6] * (p[:,:,7] - p[:,:,6])
    feats['press_imb'] = feats['press_bid'] - feats['press_ask']

    step_arr = np.tile(np.arange(SEQ_LEN, dtype=np.float32), (n_seqs, 1))
    feats['step_norm'] = step_arr

    # Free large arrays we no longer need
    del p, v, tp, tv; gc.collect()

    print(f"    Base features: {len(feats)}", flush=True)

    key_series = {
        'mid': mid, 'spread': spread, 'vol_imb': vol_imb,
        'tvol': tvol, 'imb_l0': feats['imb_l0'],
    }

    for w in [5, 10, 20, 50, 100]:
        print(f"    Rolling {w}...", flush=True)
        for col, arr in key_series.items():
            feats[f'{col}_mean_{w}'] = rolling_mean(arr, w)
            if col in ('mid', 'spread', 'vol_imb'):
                feats[f'{col}_std_{w}'] = rolling_std(arr, w)
        feats[f'mid_mom_{w}'] = rolling_diff(mid, w)

    trade_series = {
        'signed_flow': feats['signed_flow'],
        'trade_intensity': feats['trade_intensity'],
        'press_imb': feats['press_imb'],
    }
    for w in [5, 10, 20]:
        for col, arr in trade_series.items():
            feats[f'{col}_mean_{w}'] = rolling_mean(arr, w)

    print("    Lag/diff...", flush=True)
    for col, arr in key_series.items():
        for k in [1, 2, 3, 5]:
            feats[f'{col}_lag{k}'] = lag(arr, k)
        feats[f'{col}_diff1'] = arr - lag(arr, 1)
        feats[f'{col}_diff2'] = arr - lag(arr, 2)
        feats[f'{col}_diff5'] = arr - lag(arr, 5)

    print("    Pairwise diffs...", flush=True)
    for col in ['mid', 'vol_imb', 'imb_l0']:
        arr = key_series[col]
        l1, l2, l3, l5 = lag(arr,1), lag(arr,2), lag(arr,3), lag(arr,5)
        feats[f'{col}_d12'] = l1 - l2
        feats[f'{col}_d13'] = l1 - l3
        feats[f'{col}_d23'] = l2 - l3
        feats[f'{col}_d15'] = l1 - l5
        feats[f'{col}_d25'] = l2 - l5

    print("    EWM...", flush=True)
    for col in ['mid', 'vol_imb', 'imb_l0']:
        for span in [5, 20]:
            feats[f'{col}_ewm_{span}'] = ewm_mean(key_series[col], span)
            feats[f'{col}_ewm_dev_{span}'] = key_series[col] - feats[f'{col}_ewm_{span}']

    feats['spread_x_vol_imb'] = spread * vol_imb
    feats['imb_l0_x_tvol'] = feats['imb_l0'] * tvol
    feats['press_imb_x_vol_imb'] = feats['press_imb'] * vol_imb
    feats['mid_mom5_x_vol_imb'] = rolling_diff(mid, 5) * vol_imb
    feats['spread_x_imb_l0'] = spread * feats['imb_l0']
    feats['trade_flow_x_imb'] = feats['signed_flow'] * feats['imb_l0']

    feats['spread_to_depth'] = spread / (feats['bid_depth'] + feats['ask_depth'] + 1e-8)
    feats['vol_bid_to_ask_ratio'] = tvol_bid / (tvol_ask + 1e-8)
    feats['vwap_mid_diff'] = (vwap_bid + vwap_ask) / 2 - mid

    elapsed = time.time() - t0_time
    print(f"  Engineered {len(feats)} feature columns in {elapsed:.1f}s", flush=True)
    return feats, list(feats.keys())


def weighted_pearson(y_true, y_pred, weights=None):
    y_pred_clip = np.clip(y_pred, -6, 6)
    if weights is None:
        weights = np.maximum(np.abs(y_true), 1e-8)
    w_sum = np.sum(weights)
    mean_t = np.sum(y_true * weights) / w_sum
    mean_p = np.sum(y_pred_clip * weights) / w_sum
    dt = y_true - mean_t
    dp = y_pred_clip - mean_p
    cov = np.sum(weights * dt * dp) / w_sum
    var_t = np.sum(weights * dt ** 2) / w_sum
    var_p = np.sum(weights * dp ** 2) / w_sum
    if var_t <= 0 or var_p <= 0:
        return 0.0
    return cov / (np.sqrt(var_t) * np.sqrt(var_p))


def build_dataset(df, n_seqs, raw_cols, row_subsample=1.0):
    feats_dict, eng_names = engineer_features_fast(df, n_seqs)

    mask = df['need_prediction'].values.astype(bool)
    targets_t0 = df['t0'].values[mask].astype(np.float32)
    targets_t1 = df['t1'].values[mask].astype(np.float32)

    feature_names = raw_cols + eng_names
    n_pred = mask.sum()
    n_feat = len(feature_names)

    if row_subsample < 1.0:
        np.random.seed(42)
        keep = np.random.rand(n_pred) < row_subsample
        print(f"  Row subsample: keeping {keep.sum():,} of {n_pred:,} rows ({row_subsample*100:.0f}%)", flush=True)
    else:
        keep = np.ones(n_pred, dtype=bool)

    n_keep = keep.sum()
    print(f"  Building matrix: {n_keep:,} rows x {n_feat} features...", flush=True)
    X = np.empty((n_keep, n_feat), dtype=np.float32)

    raw_vals = df[raw_cols].values.astype(np.float32)
    X[:, :len(raw_cols)] = raw_vals[mask][keep]
    del raw_vals

    for i, name in enumerate(eng_names):
        arr = feats_dict[name].reshape(-1).astype(np.float32)
        X[:, len(raw_cols) + i] = arr[mask][keep]

    del feats_dict; gc.collect()
    return X, targets_t0[keep], targets_t1[keep], feature_names


def main():
    print("=" * 60, flush=True)
    print("LightGBM v2: Training on ALL sequences", flush=True)
    print("=" * 60, flush=True)

    print("\nLoading training data...", flush=True)
    train_df = pd.read_parquet(TRAIN_PATH)
    train_df = train_df.sort_values(['seq_ix', 'step_in_seq']).reset_index(drop=True)
    raw_cols = [c for c in train_df.columns if c not in ['seq_ix', 'step_in_seq', 'need_prediction', 't0', 't1']]
    n_total_seqs = train_df['seq_ix'].nunique()
    print(f"  Total: {len(train_df):,} rows, {n_total_seqs} sequences", flush=True)

    if N_TRAIN_SEQS is not None and N_TRAIN_SEQS < n_total_seqs:
        all_seqs = sorted(train_df['seq_ix'].unique())
        np.random.seed(42)
        selected = np.random.choice(all_seqs, size=N_TRAIN_SEQS, replace=False)
        train_df = train_df[train_df['seq_ix'].isin(selected)].reset_index(drop=True)
        n_seqs = N_TRAIN_SEQS
    else:
        n_seqs = n_total_seqs
    print(f"  Using: {len(train_df):,} rows, {n_seqs} sequences", flush=True)

    print("\n--- Training features ---", flush=True)
    X_train, t0_train, t1_train, feature_names = build_dataset(
        train_df, n_seqs, raw_cols, row_subsample=0.3
    )
    del train_df; gc.collect()
    print(f"  X_train: {X_train.shape}, {X_train.nbytes / 1e9:.2f} GB", flush=True)

    print("\nLoading validation data...", flush=True)
    valid_df = pd.read_parquet(VALID_PATH)
    valid_df = valid_df.sort_values(['seq_ix', 'step_in_seq']).reset_index(drop=True)
    n_valid_seqs = valid_df['seq_ix'].nunique()

    print("\n--- Validation features ---", flush=True)
    X_valid, t0_valid, t1_valid, _ = build_dataset(valid_df, n_valid_seqs, raw_cols)
    del valid_df; gc.collect()
    print(f"  X_valid: {X_valid.shape}, {X_valid.nbytes / 1e9:.2f} GB", flush=True)

    print(f"\nTotal features: {len(feature_names)}", flush=True)
    print(f"  Training samples: {X_train.shape[0]:,}", flush=True)
    print(f"  Validation samples: {X_valid.shape[0]:,}", flush=True)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 127,
        'max_depth': -1,
        'min_data_in_leaf': 200,  # more conservative with more data
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }

    results = {}

    for target_name, y_train, y_valid in [
        ('t0', t0_train, t0_valid),
        ('t1', t1_train, t1_valid),
    ]:
        print(f"\n{'='*50}", flush=True)
        print(f"Training LightGBM for {target_name}", flush=True)
        print(f"{'='*50}", flush=True)

        w_train = np.maximum(np.abs(y_train), 1e-8).astype(np.float32)
        w_valid = np.maximum(np.abs(y_valid), 1e-8).astype(np.float32)

        dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train,
                            feature_name=feature_names, free_raw_data=False)
        dvalid = lgb.Dataset(X_valid, label=y_valid, weight=w_valid,
                            feature_name=feature_names, free_raw_data=False)

        # Use larger early_stop for t1 (it tends to stop too early)
        early = 100 if target_name == 't1' else 50

        t_start = time.time()
        model = lgb.train(
            params, dtrain,
            num_boost_round=3000,
            valid_sets=[dtrain, dvalid],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early),
                lgb.log_evaluation(period=50),
            ],
        )
        elapsed = time.time() - t_start

        print(f"\nTraining time: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
        print(f"Best iteration: {model.best_iteration}", flush=True)

        pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
        wp_valid = weighted_pearson(y_valid, pred_valid)
        print(f"\nWeighted Pearson ({target_name}): {wp_valid:.6f}", flush=True)

        results[target_name] = {
            'wp_valid': wp_valid,
            'best_iter': model.best_iteration,
        }

        # Save model
        model_path = WORK_DIR / f"lgbm_v2_{target_name}.txt"
        model.save_model(str(model_path), num_iteration=model.best_iteration)
        model_size = os.path.getsize(model_path) / 1024 / 1024
        print(f"  Model saved: {model_path} ({model_size:.2f} MB)", flush=True)

        # Top 20 features
        imp = model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:20]
        print(f"\n  Top 20 features ({target_name}):", flush=True)
        for i, idx in enumerate(top_idx):
            print(f"    {i+1:2d}. {feature_names[idx]:30s} gain={imp[idx]:.0f}", flush=True)

    wp_t0 = results['t0']['wp_valid']
    wp_t1 = results['t1']['wp_valid']
    combined = (wp_t0 + wp_t1) / 2

    print(f"\n{'='*60}", flush=True)
    print(f"FINAL VALIDATION SCORES (v2 - full data)", flush=True)
    print(f"  t0: {wp_t0:.6f}", flush=True)
    print(f"  t1: {wp_t1:.6f}", flush=True)
    print(f"  Combined: {combined:.6f}", flush=True)
    print(f"{'='*60}", flush=True)
    print("Done!", flush=True)


if __name__ == '__main__':
    main()
