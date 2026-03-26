"""Quick ONNX export from saved checkpoint"""
import torch, torch.nn as nn, numpy as np, os, sys
sys.stdout.reconfigure(line_buffering=True)

class DualBiGRU_CNN(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=1, dropout=0.15):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.price_proj = nn.Sequential(nn.Linear(37, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.price_gru = nn.GRU(hidden_dim, hidden_dim//2, num_layers=num_layers, batch_first=True,
                                 dropout=dropout if num_layers>1 else 0, bidirectional=True)
        self.price_norm = nn.LayerNorm(hidden_dim)
        self.volume_proj = nn.Sequential(nn.Linear(37, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.volume_gru = nn.GRU(hidden_dim, hidden_dim//2, num_layers=num_layers, batch_first=True,
                                  dropout=dropout if num_layers>1 else 0, bidirectional=True)
        self.volume_norm = nn.LayerNorm(hidden_dim)
        cat_dim = hidden_dim * 2
        self.cnn = nn.Sequential(
            nn.Conv1d(cat_dim, cat_dim, 3, padding=1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(cat_dim, hidden_dim, 5, padding=2), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim//2, 7, padding=3), nn.GELU(),
        )
        self.cnn_norm = nn.LayerNorm(hidden_dim//2)
        self.head = nn.Sequential(nn.Linear(hidden_dim//2, hidden_dim//4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim//4, 2))

    def forward(self, x):
        price_raw = x[:,:,:12]; vol_raw = x[:,:,12:24]
        price_trade = x[:,:,24:28]; vol_trade = x[:,:,28:32]; shared_eng = x[:,:,32:]
        x_price = torch.cat([price_raw, price_trade, shared_eng], dim=-1)
        p = self.price_proj(x_price); p, _ = self.price_gru(p); p = self.price_norm(p)
        x_vol = torch.cat([vol_raw, vol_trade, shared_eng], dim=-1)
        v = self.volume_proj(x_vol); v, _ = self.volume_gru(v); v = self.volume_norm(v)
        cat = torch.cat([p, v], dim=-1)
        cnn_out = self.cnn(cat.transpose(1,2)).transpose(1,2)
        cnn_out = self.cnn_norm(cnn_out)
        return self.head(cnn_out)

print('Loading...', flush=True)
model = DualBiGRU_CNN(hidden_dim=64, num_layers=1)
state = torch.load('best_dual_bigru_cnn.pt', map_location='cpu', weights_only=True)
model.load_state_dict(state)
model.eval()
print('Loaded OK', flush=True)

dummy = torch.randn(1, 100, 53)
onnx_path = 'model_dual_bigru_cnn.onnx'
try:
    torch.onnx.export(model, dummy, onnx_path, export_params=True, opset_version=14,
        do_constant_folding=True, input_names=['input'], output_names=['output'],
        dynamic_axes={'input':{0:'batch',1:'seq_len'},'output':{0:'batch',1:'seq_len'}}, dynamo=False)
except TypeError:
    torch.onnx.export(model, dummy, onnx_path, export_params=True, opset_version=14,
        do_constant_folding=True, input_names=['input'], output_names=['output'],
        dynamic_axes={'input':{0:'batch',1:'seq_len'},'output':{0:'batch',1:'seq_len'}})

print(f'ONNX: {os.path.getsize(onnx_path)} bytes', flush=True)
import onnxruntime as ort
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
out = sess.run(None, {'input': dummy.numpy()})
print(f'Verify: shape={out[0].shape}', flush=True)
print('SUCCESS', flush=True)
