"""
Dual BiGRU + CNN1d solution for Wunder Fund Predictorium.

Architecture: Two BiGRU branches (price + volume) with shared engineered features,
fused through CNN1d stack.

Inference: Batch mode - buffer all raw features, periodically run full model.
"""
import os
import numpy as np
import onnxruntime as ort
from typing import Optional


class PredictionModel:
    """Dual BiGRU + CNN1d with batch inference."""

    CACHE_INTERVAL = 100  # Re-run inference every N steps

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            os.path.join(base_dir, "model_dual_bigru_cnn.onnx"),
            sess_options, providers=['CPUExecutionProvider']
        )

        stats = np.load(os.path.join(base_dir, "dual_bigru_cnn_norm_stats.npz"))
        self.mean = stats['mean'].astype(np.float32)
        self.std = stats['std'].astype(np.float32)

        self.current_seq_ix = None
        self.raw_buffer = []
        self.cached_preds = None
        self.last_cache_step = -1

    def _reset(self):
        self.raw_buffer = []
        self.cached_preds = None
        self.last_cache_step = -1

    def predict(self, data_point) -> Optional[np.ndarray]:
        if data_point.seq_ix != self.current_seq_ix:
            self._reset()
            self.current_seq_ix = data_point.seq_ix

        self.raw_buffer.append(data_point.state.astype(np.float32))
        step = len(self.raw_buffer) - 1

        if not data_point.need_prediction:
            return None

        # Refresh cache if stale or first prediction
        if self.cached_preds is None or step - self.last_cache_step >= self.CACHE_INTERVAL:
            self._run_inference()

        pred = self.cached_preds[step]
        return np.clip(pred, -6.0, 6.0)

    def _run_inference(self):
        raw = np.array(self.raw_buffer, dtype=np.float32)  # (n_steps, 32)
        features = self._engineer_features_batch(raw)       # (n_steps, 53)
        norm = ((features - self.mean) / self.std).astype(np.float32)
        input_tensor = norm.reshape(1, -1, norm.shape[1])   # (1, n_steps, 53)
        output = self.session.run(None, {'input': input_tensor})[0]
        self.cached_preds = output[0]  # (n_steps, 2)
        self.last_cache_step = len(self.raw_buffer) - 1

    @staticmethod
    def _engineer_features_batch(raw_features):
        """Vectorized feature engineering. (n, 32) -> (n, 53)"""
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
        pmom5 = np.zeros(N, dtype=np.float32)
        pvol5 = np.zeros(N, dtype=np.float32)
        pmom20 = np.zeros(N, dtype=np.float32)
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
