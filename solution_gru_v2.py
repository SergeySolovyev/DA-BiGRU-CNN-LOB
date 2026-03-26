import os
import numpy as np
import onnxruntime as ort
from typing import Optional


class PredictionModel:
    """GRU v2: same architecture as v1 (hidden=128, 2 layers) but with 200+ features."""

    HIDDEN_DIM = 128
    NUM_LAYERS = 2

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        onnx_path = os.path.join(base_dir, "model_gru_v2_incr.onnx")
        stats_path = os.path.join(base_dir, "gru_v2_norm_stats.npz")

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            onnx_path, sess_options, providers=['CPUExecutionProvider']
        )

        stats = np.load(stats_path)
        self.feat_mean = stats['mean'].astype(np.float32)
        self.feat_std = stats['std'].astype(np.float32)

        self.current_seq_ix = None
        self.hidden = None
        self.step = 0

        # History buffers (plain lists, grow up to 1000)
        self.key_hist = {}
        self.trade_hist = {}
        self.ewm_states = {}

    def _reset(self):
        self.step = 0
        self.hidden = np.zeros((self.NUM_LAYERS, 1, self.HIDDEN_DIM), dtype=np.float32)
        self.key_hist = {name: [] for name in ['mid', 'spread', 'vol_imb', 'tvol', 'imb_l0']}
        self.trade_hist = {name: [] for name in ['signed_flow', 'trade_intensity', 'press_imb']}
        self.ewm_states = {}

    def predict(self, data_point) -> Optional[np.ndarray]:
        if data_point.seq_ix != self.current_seq_ix:
            self._reset()
            self.current_seq_ix = data_point.seq_ix

        raw = data_point.state
        features = self._compute_features(raw)
        norm = (features - self.feat_mean) / self.feat_std

        x_step = norm.reshape(1, 1, -1).astype(np.float32)
        pred_out, self.hidden = self.session.run(
            None, {'x_step': x_step, 'hidden_in': self.hidden}
        )

        self.step += 1

        if not data_point.need_prediction:
            return None

        return np.clip(pred_out[0, 0], -6.0, 6.0)

    def _compute_features(self, raw):
        p = raw[:12]
        v = raw[12:24]
        tp = raw[24:28]
        tv = raw[28:32]

        # ---- Base features (56 total, exact order as training) ----
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
            mid, spread, rel_spread, wmid, wmid_diff,
            *imb,
            tvol_bid, tvol_ask, tvol, vol_imb, log_tvol,
            vol_conc_bid, vol_conc_ask, vol_conc_bid2, vol_conc_ask2,
            bid_depth, ask_depth, depth_imb,
            *gaps,
            avg_tp, avg_tv, trade_imb_feat, trade_intensity, signed_flow, tp_vs_mid,
            vwap_bid, vwap_ask, vwap_spread, log_v0, log_v6,
            press_bid, press_ask, press_imb,
            spread_x_vol_imb, imb_l0_x_tvol, press_imb_x_vol_imb,
            spread_x_imb_l0, trade_flow_x_imb,
            spread_to_depth, vol_ratio, vwap_mid_diff,
            step_norm,
        ]

        # ---- Update history buffers ----
        key_vals = {
            'mid': mid, 'spread': spread, 'vol_imb': vol_imb,
            'tvol': tvol, 'imb_l0': imb[0]
        }
        trade_vals = {
            'signed_flow': signed_flow, 'trade_intensity': trade_intensity,
            'press_imb': press_imb
        }
        for name, val in key_vals.items():
            self.key_hist[name].append(float(val))
        for name, val in trade_vals.items():
            self.trade_hist[name].append(float(val))

        # ---- Rolling features (64 total) ----
        # Pre-convert to numpy arrays for efficient slicing
        key_np = {name: np.array(hist, dtype=np.float32) for name, hist in self.key_hist.items()}
        trade_np = {name: np.array(hist, dtype=np.float32) for name, hist in self.trade_hist.items()}

        rolling_feats = []
        for w in [5, 10, 20, 50, 100]:
            for name in ['mid', 'spread', 'vol_imb', 'tvol', 'imb_l0']:
                arr = key_np[name]
                n = len(arr)
                if n >= w:
                    window = arr[-w:]
                else:
                    window = arr
                mean_val = float(np.mean(window))
                mean_sq = float(np.mean(window ** 2))
                std_val = max(mean_sq - mean_val ** 2, 0) ** 0.5
                rolling_feats.append(mean_val)
                rolling_feats.append(std_val)
            # Mid momentum
            mid_arr = key_np['mid']
            if len(mid_arr) > w:
                rolling_feats.append(float(mid_arr[-1] - mid_arr[-(w+1)]))
            else:
                rolling_feats.append(0.0)

        # Trade series rolling means
        for w in [5, 10, 20]:
            for name in ['signed_flow', 'trade_intensity', 'press_imb']:
                arr = trade_np[name]
                n = len(arr)
                if n >= w:
                    rolling_feats.append(float(np.mean(arr[-w:])))
                else:
                    rolling_feats.append(float(np.mean(arr)))

        # ---- Lag/diff features (55 total) ----
        lag_diff_feats = []
        for name in ['mid', 'spread', 'vol_imb', 'tvol', 'imb_l0']:
            hist = self.key_hist[name]
            current = hist[-1]
            for k in [1, 2, 3, 5]:
                if len(hist) > k:
                    lagged = hist[-(k+1)]
                else:
                    lagged = 0.0
                lag_diff_feats.append(lagged)
                lag_diff_feats.append(current - lagged)

        # Pairwise lag diffs
        for name in ['mid', 'vol_imb', 'imb_l0']:
            hist = self.key_hist[name]
            lags = {}
            for k in [1, 2, 3, 5]:
                if len(hist) > k:
                    lags[k] = hist[-(k+1)]
                else:
                    lags[k] = 0.0
            lag_diff_feats.extend([
                lags[1] - lags[2], lags[1] - lags[3], lags[2] - lags[3],
                lags[1] - lags[5], lags[2] - lags[5],
            ])

        # ---- EWM features (12 total) ----
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

        # ---- Combine: raw(32) + base(56) + rolling(64) + lag_diff(55) + ewm(12) = 219 ----
        engineered = np.array(base + rolling_feats + lag_diff_feats + ewm_feats, dtype=np.float32)
        return np.concatenate([raw.astype(np.float32), engineered])
