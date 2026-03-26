"""
3-Model Ensemble: GRU v1 (incremental) + GRU v2 (incremental) + Dual BiGRU-CNN (batch).

- GRU v1: 53 features, step-by-step hidden state
- GRU v2: 219 features, step-by-step hidden state
- Dual BiGRU-CNN: 53 features, batch inference with periodic caching

Final prediction: weighted average of all 3 models.
"""
import os
import numpy as np
import onnxruntime as ort
from collections import deque
from typing import Optional


class PredictionModel:
    """3-model ensemble with mixed inference strategies."""

    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    W1 = 0.35   # GRU v1 (proven on test: 0.2662)
    W2 = 0.30   # GRU v2 (219 features, validation 0.2476)
    W3 = 0.35   # Dual BiGRU-CNN (new architecture)
    CACHE_INTERVAL = 100  # for BiGRU-CNN batch inference

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # --- GRU v1 (53 features, incremental) ---
        self.sess_v1 = ort.InferenceSession(
            os.path.join(base_dir, "model_gru_incr.onnx"),
            sess_options, providers=['CPUExecutionProvider']
        )
        stats_v1 = np.load(os.path.join(base_dir, "gru_norm_stats.npz"))
        self.mean_v1 = stats_v1['mean'].astype(np.float32)
        self.std_v1 = stats_v1['std'].astype(np.float32)

        # --- GRU v2 (219 features, incremental) ---
        self.sess_v2 = ort.InferenceSession(
            os.path.join(base_dir, "model_gru_v2_incr.onnx"),
            sess_options, providers=['CPUExecutionProvider']
        )
        stats_v2 = np.load(os.path.join(base_dir, "gru_v2_norm_stats.npz"))
        self.mean_v2 = stats_v2['mean'].astype(np.float32)
        self.std_v2 = stats_v2['std'].astype(np.float32)

        # --- Dual BiGRU-CNN (53 features, batch mode) ---
        self.sess_cnn = ort.InferenceSession(
            os.path.join(base_dir, "model_dual_bigru_cnn.onnx"),
            sess_options, providers=['CPUExecutionProvider']
        )
        stats_cnn = np.load(os.path.join(base_dir, "dual_bigru_cnn_norm_stats.npz"))
        self.mean_cnn = stats_cnn['mean'].astype(np.float32)
        self.std_cnn = stats_cnn['std'].astype(np.float32)

        self.current_seq_ix = None
        self.hidden_v1 = None
        self.hidden_v2 = None
        self.step = 0

        # v1 buffers
        self.mid_buffer = deque(maxlen=21)

        # v2 buffers
        self.key_hist = {}
        self.trade_hist = {}
        self.ewm_states = {}

        # BiGRU-CNN batch buffers
        self.raw_buffer = []
        self.cached_cnn_preds = None
        self.last_cache_step = -1

    def _reset(self):
        self.step = 0
        self.hidden_v1 = np.zeros((self.NUM_LAYERS, 1, self.HIDDEN_DIM), dtype=np.float32)
        self.hidden_v2 = np.zeros((self.NUM_LAYERS, 1, self.HIDDEN_DIM), dtype=np.float32)
        self.mid_buffer.clear()
        self.key_hist = {name: [] for name in ['mid', 'spread', 'vol_imb', 'tvol', 'imb_l0']}
        self.trade_hist = {name: [] for name in ['signed_flow', 'trade_intensity', 'press_imb']}
        self.ewm_states = {}
        self.raw_buffer = []
        self.cached_cnn_preds = None
        self.last_cache_step = -1

    def predict(self, data_point) -> Optional[np.ndarray]:
        if data_point.seq_ix != self.current_seq_ix:
            self._reset()
            self.current_seq_ix = data_point.seq_ix

        raw = data_point.state
        self.raw_buffer.append(raw.astype(np.float32))

        # --- V1 inference (incremental) ---
        feat_v1 = self._features_v1(raw)
        norm_v1 = ((feat_v1 - self.mean_v1) / self.std_v1).reshape(1, 1, -1).astype(np.float32)
        pred_v1, self.hidden_v1 = self.sess_v1.run(
            None, {'x_step': norm_v1, 'hidden_in': self.hidden_v1}
        )

        # --- V2 inference (incremental) ---
        feat_v2 = self._features_v2(raw)
        norm_v2 = ((feat_v2 - self.mean_v2) / self.std_v2).reshape(1, 1, -1).astype(np.float32)
        pred_v2, self.hidden_v2 = self.sess_v2.run(
            None, {'x_step': norm_v2, 'hidden_in': self.hidden_v2}
        )

        self.step += 1

        if not data_point.need_prediction:
            return None

        # --- BiGRU-CNN inference (batch, periodic) ---
        step_idx = len(self.raw_buffer) - 1
        if self.cached_cnn_preds is None or step_idx - self.last_cache_step >= self.CACHE_INTERVAL:
            self._run_cnn_inference()
        p3 = self.cached_cnn_preds[step_idx]

        p1 = pred_v1[0, 0]
        p2 = pred_v2[0, 0]
        pred = self.W1 * p1 + self.W2 * p2 + self.W3 * p3
        return np.clip(pred, -6.0, 6.0)

    # =====================================================================
    # BiGRU-CNN batch inference
    # =====================================================================
    def _run_cnn_inference(self):
        raw = np.array(self.raw_buffer, dtype=np.float32)
        features = self._engineer_features_batch_v1(raw)
        norm = ((features - self.mean_cnn) / self.std_cnn).astype(np.float32)
        input_tensor = norm.reshape(1, -1, norm.shape[1])
        output = self.sess_cnn.run(None, {'input': input_tensor})[0]
        self.cached_cnn_preds = output[0]
        self.last_cache_step = len(self.raw_buffer) - 1

    @staticmethod
    def _engineer_features_batch_v1(raw_features):
        """Vectorized v1 features. (n, 32) -> (n, 53)"""
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

    # =====================================================================
    # V1 features (53 dims: 32 raw + 21 engineered)
    # =====================================================================
    def _features_v1(self, raw):
        p = raw[:12]
        v = raw[12:24]
        tp = raw[24:28]
        tv = raw[28:32]

        mid = (p[0] + p[6]) / 2
        spread = p[6] - p[0]
        wmid = (p[0] * v[6] + p[6] * v[0]) / (v[0] + v[6] + 1e-8)
        imb_top = (v[0] - v[6]) / (v[0] + v[6] + 1e-8)
        tvol_bid = np.sum(v[:6])
        tvol_ask = np.sum(v[6:])
        tvol = tvol_bid + tvol_ask
        vol_imb = (tvol_bid - tvol_ask) / (tvol + 1e-8)
        rel_spread = spread / (mid + 1e-8)

        imb_l2 = (v[1] - v[7]) / (v[1] + v[7] + 1e-8)
        imb_l3 = (v[2] - v[8]) / (v[2] + v[8] + 1e-8)
        bid_depth = p[0] - p[5]
        ask_depth = p[11] - p[6]
        avg_tp = np.mean(tp)
        avg_tv = np.mean(tv)
        trade_imb = (tp[0] - tp[1]) / (abs(tp[0]) + abs(tp[1]) + 1e-8)
        vol_conc_bid = v[0] / (tvol_bid + 1e-8)
        vol_conc_ask = v[6] / (tvol_ask + 1e-8)
        log_tvol = np.log1p(abs(tvol))
        log_v0 = np.log1p(abs(v[0]))

        self.mid_buffer.append(mid)
        n_buf = len(self.mid_buffer)

        if n_buf >= 6:
            buf = list(self.mid_buffer)
            pmom5 = buf[-1] - buf[-6]
            window = buf[-6:]
            mean_w = sum(window) / 6.0
            pvol5 = (sum((x - mean_w) ** 2 for x in window) / 6.0) ** 0.5
        else:
            pmom5 = 0.0
            pvol5 = 0.0

        pmom20 = (self.mid_buffer[-1] - self.mid_buffer[0]) if n_buf >= 21 else 0.0

        eng = np.array([
            mid, spread, wmid, imb_top, tvol, vol_imb, rel_spread,
            imb_l2, imb_l3, bid_depth, ask_depth,
            avg_tp, avg_tv, trade_imb,
            vol_conc_bid, vol_conc_ask, log_tvol, log_v0,
            pmom5, pvol5, pmom20
        ], dtype=np.float32)
        return np.concatenate([raw.astype(np.float32), eng])

    # =====================================================================
    # V2 features (219 dims: 32 raw + 187 engineered)
    # =====================================================================
    def _features_v2(self, raw):
        p = raw[:12]
        v = raw[12:24]
        tp = raw[24:28]
        tv = raw[28:32]

        mid = (p[0] + p[6]) / 2
        spread = p[6] - p[0]
        rel_spread = spread / (abs(mid) + 1e-8)
        wmid = (p[0] * v[6] + p[6] * v[0]) / (v[0] + v[6] + 1e-8)
        wmid_diff = wmid - mid
        imb = [(v[i] - v[6+i]) / (v[i] + v[6+i] + 1e-8) for i in range(6)]

        tvol_bid = float(np.sum(v[:6]))
        tvol_ask = float(np.sum(v[6:]))
        tvol = tvol_bid + tvol_ask
        vol_imb = (tvol_bid - tvol_ask) / (tvol + 1e-8)
        log_tvol = float(np.log1p(abs(tvol)))
        vol_conc_bid = v[0] / (tvol_bid + 1e-8)
        vol_conc_ask = v[6] / (tvol_ask + 1e-8)
        vol_conc_bid2 = (v[0] + v[1]) / (tvol_bid + 1e-8)
        vol_conc_ask2 = (v[6] + v[7]) / (tvol_ask + 1e-8)

        bid_depth = p[0] - p[5]
        ask_depth = p[11] - p[6]
        depth_imb = bid_depth - ask_depth

        gaps = []
        for i in range(5):
            gaps.append(p[i] - p[i+1])
            gaps.append(p[6+i+1] - p[6+i])

        avg_tp = float(np.mean(tp))
        avg_tv = float(np.mean(tv))
        trade_imb_feat = (tp[0] - tp[1]) / (abs(tp[0]) + abs(tp[1]) + 1e-8)
        trade_intensity = float(np.sum(np.abs(tv)))
        signed_flow = float(np.sum(tp * tv))
        tp_vs_mid = avg_tp - mid

        vwap_bid = float(np.sum(p[:6] * v[:6])) / (tvol_bid + 1e-8)
        vwap_ask = float(np.sum(p[6:] * v[6:])) / (tvol_ask + 1e-8)
        vwap_spread = vwap_ask - vwap_bid
        log_v0 = float(np.log1p(abs(v[0])))
        log_v6 = float(np.log1p(abs(v[6])))

        press_bid = v[0] * (p[0] - p[1])
        press_ask = v[6] * (p[7] - p[6])
        press_imb = press_bid - press_ask

        spread_x_vol_imb = spread * vol_imb
        imb_l0_x_tvol = imb[0] * tvol
        press_imb_x_vol_imb = press_imb * vol_imb
        spread_x_imb_l0 = spread * imb[0]
        trade_flow_x_imb = signed_flow * imb[0]
        spread_to_depth = spread / (bid_depth + ask_depth + 1e-8)
        vol_ratio = tvol_bid / (tvol_ask + 1e-8)
        vwap_mid_diff = (vwap_bid + vwap_ask) / 2 - mid
        step_norm = self.step / 999.0

        base = [
            mid, spread, rel_spread, wmid, wmid_diff, *imb,
            tvol_bid, tvol_ask, tvol, vol_imb, log_tvol,
            vol_conc_bid, vol_conc_ask, vol_conc_bid2, vol_conc_ask2,
            bid_depth, ask_depth, depth_imb, *gaps,
            avg_tp, avg_tv, trade_imb_feat, trade_intensity, signed_flow, tp_vs_mid,
            vwap_bid, vwap_ask, vwap_spread, log_v0, log_v6,
            press_bid, press_ask, press_imb,
            spread_x_vol_imb, imb_l0_x_tvol, press_imb_x_vol_imb,
            spread_x_imb_l0, trade_flow_x_imb,
            spread_to_depth, vol_ratio, vwap_mid_diff, step_norm,
        ]

        # Update history
        key_vals = {'mid': mid, 'spread': spread, 'vol_imb': vol_imb,
                    'tvol': tvol, 'imb_l0': imb[0]}
        trade_vals = {'signed_flow': signed_flow, 'trade_intensity': trade_intensity,
                      'press_imb': press_imb}
        for name, val in key_vals.items():
            self.key_hist[name].append(float(val))
        for name, val in trade_vals.items():
            self.trade_hist[name].append(float(val))

        # Rolling features
        key_np = {n: np.array(h, dtype=np.float32) for n, h in self.key_hist.items()}
        trade_np = {n: np.array(h, dtype=np.float32) for n, h in self.trade_hist.items()}

        rolling_feats = []
        for w in [5, 10, 20, 50, 100]:
            for name in ['mid', 'spread', 'vol_imb', 'tvol', 'imb_l0']:
                arr = key_np[name]
                n = len(arr)
                window = arr[-w:] if n >= w else arr
                mv = float(np.mean(window))
                msq = float(np.mean(window ** 2))
                rolling_feats.append(mv)
                rolling_feats.append(max(msq - mv * mv, 0) ** 0.5)
            mid_arr = key_np['mid']
            rolling_feats.append(float(mid_arr[-1] - mid_arr[-(w+1)]) if len(mid_arr) > w else 0.0)

        for w in [5, 10, 20]:
            for name in ['signed_flow', 'trade_intensity', 'press_imb']:
                arr = trade_np[name]
                rolling_feats.append(float(np.mean(arr[-w:])) if len(arr) >= w else float(np.mean(arr)))

        # Lag/diff features
        lag_diff_feats = []
        for name in ['mid', 'spread', 'vol_imb', 'tvol', 'imb_l0']:
            hist = self.key_hist[name]
            current = hist[-1]
            for k in [1, 2, 3, 5]:
                lagged = hist[-(k+1)] if len(hist) > k else 0.0
                lag_diff_feats.append(lagged)
                lag_diff_feats.append(current - lagged)

        for name in ['mid', 'vol_imb', 'imb_l0']:
            hist = self.key_hist[name]
            lags = {k: (hist[-(k+1)] if len(hist) > k else 0.0) for k in [1, 2, 3, 5]}
            lag_diff_feats.extend([
                lags[1] - lags[2], lags[1] - lags[3], lags[2] - lags[3],
                lags[1] - lags[5], lags[2] - lags[5],
            ])

        # EWM features
        ewm_feats = []
        for name in ['mid', 'vol_imb', 'imb_l0']:
            current = key_vals[name]
            for span in [5, 20]:
                alpha = 2.0 / (span + 1.0)
                key = (name, span)
                if key not in self.ewm_states:
                    self.ewm_states[key] = float(current)
                else:
                    self.ewm_states[key] = alpha * float(current) + (1 - alpha) * self.ewm_states[key]
                ewm_feats.append(self.ewm_states[key])
                ewm_feats.append(float(current) - self.ewm_states[key])

        engineered = np.array(base + rolling_feats + lag_diff_feats + ewm_feats, dtype=np.float32)
        return np.concatenate([raw.astype(np.float32), engineered])
