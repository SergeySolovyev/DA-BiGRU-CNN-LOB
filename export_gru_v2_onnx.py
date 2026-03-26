"""Export GRU v2 ONNX from saved checkpoint."""
import torch
import torch.nn as nn
import numpy as np
import os

WORK_DIR = r"D:\Wunder Fund\Claude"
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.1

# Load normalization stats to get input_dim
stats = np.load(os.path.join(WORK_DIR, "gru_v2_norm_stats.npz"))
INPUT_DIM = len(stats['mean'])
print(f"Input dim: {INPUT_DIM}")

# Model definition (must match training)
class CompetitiveGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.1):
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

# Incremental wrapper
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

# Load checkpoint
model = CompetitiveGRU(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
model.load_state_dict(torch.load(os.path.join(WORK_DIR, "best_gru_v2.pt"), map_location='cpu', weights_only=True))
model.eval()
print("Checkpoint loaded")

# Export
incr_model = IncrGRU(model)
incr_model.eval()

x_step = torch.randn(1, 1, INPUT_DIM)
h_in = torch.zeros(NUM_LAYERS, 1, HIDDEN_DIM)

onnx_path = os.path.join(WORK_DIR, "model_gru_v2_incr.onnx")
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
    dynamo=False,
)
onnx_size = os.path.getsize(onnx_path) / 1024
print(f"ONNX saved: {onnx_path} ({onnx_size:.1f} KB)")

# Verify
import onnxruntime as ort
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
with torch.no_grad():
    pt_pred, pt_h = incr_model(x_step, h_in)
onnx_pred, onnx_h = sess.run(None, {
    'x_step': x_step.numpy(),
    'hidden_in': h_in.numpy()
})
max_diff = max(np.max(np.abs(pt_pred.numpy() - onnx_pred)),
               np.max(np.abs(pt_h.numpy() - onnx_h)))
print(f"ONNX verification max diff: {max_diff:.2e}")
print("Done!")
