import os
import numpy as np
import onnxruntime as ort
from collections import deque
from typing import Optional


class PredictionModel:
    """GRU model with incremental single-step inference (exact predictions)"""

    HIDDEN_DIM = 128
    NUM_LAYERS = 2

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        onnx_path = os.path.join(base_dir, "model_gru_incr.onnx")
        stats_path = os.path.join(base_dir, "gru_norm_stats.npz")

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            onnx_path, sess_options, providers=['CPUExecutionProvider']
        )

        stats = np.load(stats_path)
        self.feature_mean = stats['mean'].astype(np.float32)
        self.feature_std = stats['std'].astype(np.float32)

        self.current_seq_ix = None
        self.mid_buffer = deque(maxlen=21)
        self.hidden = None  # GRU hidden state

    def predict(self, data_point) -> Optional[np.ndarray]:
        if data_point.seq_ix != self.current_seq_ix:
            self._reset_state()
            self.current_seq_ix = data_point.seq_ix

        raw = data_point.state
        eng = self._engineer_features(raw)
        norm = (eng - self.feature_mean) / self.feature_std

        # Run single-step ONNX inference (updates hidden state)
        x_step = norm.reshape(1, 1, -1).astype(np.float32)
        pred_out, self.hidden = self.session.run(
            None, {'x_step': x_step, 'hidden_in': self.hidden}
        )

        if not data_point.need_prediction:
            return None

        pred = pred_out[0, 0]  # (2,)
        return np.clip(pred, -6.0, 6.0)

    def _engineer_features(self, raw):
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

        if n_buf >= 21:
            pmom20 = self.mid_buffer[-1] - self.mid_buffer[0]
        else:
            pmom20 = 0.0

        eng = np.array([
            mid, spread, wmid, imb_top,
            tvol, vol_imb, rel_spread,
            imb_l2, imb_l3,
            bid_depth, ask_depth,
            avg_tp, avg_tv, trade_imb,
            vol_conc_bid, vol_conc_ask,
            log_tvol, log_v0,
            pmom5, pvol5, pmom20
        ], dtype=np.float32)

        return np.concatenate([raw, eng])

    def _reset_state(self):
        self.mid_buffer.clear()
        self.hidden = np.zeros(
            (self.NUM_LAYERS, 1, self.HIDDEN_DIM), dtype=np.float32
        )
