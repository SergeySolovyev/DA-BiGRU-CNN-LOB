"""
Wunder LOB Predictorium - Complete Solution Pipeline
=====================================================

End-to-end ML pipeline for predicting LOB movements
- Model: CNN + BiLSTM + Attention
- Features: 32 raw + 16 engineered
- Export: ONNX for fast inference
- Creates 3 submission variants automatically
"""

# Fix Windows console encoding
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ============================================================================
# 1. IMPORTS AND SETUP
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import zipfile
import shutil
from collections import deque
from typing import Optional, Tuple, List
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

import onnx
import onnxruntime as ort
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Configure plotting
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("✓ All imports successful")
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")


# ============================================================================
# 2. CONFIGURATION
# ============================================================================

STARTER_PACK_PATH = Path(r"d:\Wunder Fund\Codex\wnn_predictorium_starterpack.zip")
WORK_DIR = Path(r"d:\Wunder Fund\Claude")
DATA_DIR = WORK_DIR / "datasets"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
(WORK_DIR / "weights").mkdir(exist_ok=True)
(WORK_DIR / "submission").mkdir(exist_ok=True)

# Model configurations for different submissions
MODEL_CONFIGS = {
    'baseline': {
        'name': 'LOBNet_Baseline',
        'lstm_hidden_dim': 128,
        'lstm_layers': 2,
        'attention_heads': 4,
        'epochs': 15,
        'lr': 1e-3
    },
    'large': {
        'name': 'LOBNet_Large',
        'lstm_hidden_dim': 256,
        'lstm_layers': 3,
        'attention_heads': 8,
        'epochs': 20,
        'lr': 8e-4
    },
    'fast': {
        'name': 'LOBNet_Fast',
        'lstm_hidden_dim': 96,
        'lstm_layers': 2,
        'attention_heads': 4,
        'epochs': 12,
        'lr': 1.2e-3
    }
}


# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """Feature engineering for LOB data - transforms 32 raw features into 48 enriched features"""

    def __init__(self):
        self.feature_mean = None
        self.feature_std = None

    def engineer_features(self, raw_features: np.ndarray, buffer: Optional[deque] = None) -> np.ndarray:
        """Create derived features from raw LOB state"""
        # Extract components
        prices = raw_features[:12]
        volumes = raw_features[12:24]
        trade_prices = raw_features[24:28]
        trade_volumes = raw_features[28:32]

        # Price features
        mid_price = (prices[0] + prices[1]) / 2
        spread = prices[1] - prices[0]
        weighted_mid = (prices[0] * volumes[1] + prices[1] * volumes[0]) / (volumes[0] + volumes[1] + 1e-8)
        price_imbalance = (volumes[0] - volumes[1]) / (volumes[0] + volumes[1] + 1e-8)
        relative_spread = spread / (mid_price + 1e-8)
        price_depth_diff = prices[0] - prices[2]

        # Volume features
        total_volume = np.sum(volumes)
        volume_concentration = volumes[0] / (total_volume + 1e-8)
        volume_ratio = volumes[0] / (volumes[2] + 1e-8)
        log_volume = np.log1p(total_volume)

        # Trade features
        avg_trade_price = np.mean(trade_prices)
        avg_trade_volume = np.mean(trade_volumes)

        # Temporal features
        if buffer and len(buffer) >= 5:
            recent_states = np.array(list(buffer))
            recent_mid_prices = (recent_states[-5:, 0] + recent_states[-5:, 1]) / 2
            price_momentum = np.mean(np.diff(recent_mid_prices))
            volume_volatility = np.std(recent_states[-5:, 12])
        else:
            price_momentum = 0.0
            volume_volatility = 0.0

        # Combine all engineered features
        engineered = np.array([
            mid_price, spread, weighted_mid, price_imbalance,
            total_volume, volume_concentration, price_momentum, avg_trade_price,
            relative_spread, price_depth_diff, volume_ratio, log_volume,
            avg_trade_volume, np.std(volumes), volume_volatility,
            np.max(volumes) - np.min(volumes)
        ])

        return np.concatenate([raw_features, engineered])

    def fit_normalization(self, data_df: pd.DataFrame):
        """Calculate normalization statistics from training data"""
        feature_cols = [f'p{i}' for i in range(12)] + [f'v{i}' for i in range(12)] + \
                       [f'dp{i}' for i in range(4)] + [f'dv{i}' for i in range(4)]

        all_engineered = []
        for seq_ix in tqdm(data_df['seq_ix'].unique()[:1000], desc="Fitting normalization"):
            seq_data = data_df[data_df['seq_ix'] == seq_ix]
            warmup_data = seq_data[seq_data['step_in_seq'] < 99][feature_cols].values

            buffer = deque(maxlen=20)
            for raw_features in warmup_data:
                buffer.append(raw_features)
                engineered = self.engineer_features(raw_features, buffer)
                all_engineered.append(engineered)

        all_engineered = np.array(all_engineered)
        self.feature_mean = np.mean(all_engineered, axis=0)
        self.feature_std = np.std(all_engineered, axis=0) + 1e-8

        print(f"✓ Normalization fitted on {len(all_engineered)} samples")

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Standardize features using training statistics"""
        return (features - self.feature_mean) / self.feature_std


# ============================================================================
# 4. DATASET
# ============================================================================

class LOBDataset(Dataset):
    """PyTorch Dataset for LOB sequences"""

    def __init__(self, data_df: pd.DataFrame, feature_engineer: FeatureEngineer, mode='train'):
        self.data_df = data_df
        self.feature_engineer = feature_engineer
        self.mode = mode

        self.sequences = data_df['seq_ix'].unique()
        self.feature_cols = [f'p{i}' for i in range(12)] + [f'v{i}' for i in range(12)] + \
                           [f'dp{i}' for i in range(4)] + [f'dv{i}' for i in range(4)]
        self.target_cols = ['t0', 't1']

        print(f"✓ {mode.capitalize()} Dataset: {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_ix = self.sequences[idx]
        seq_data = self.data_df[self.data_df['seq_ix'] == seq_ix].sort_values('step_in_seq')

        raw_features = seq_data[self.feature_cols].values
        targets = seq_data[self.target_cols].values

        # Engineer features with buffer
        engineered_features = []
        buffer = deque(maxlen=20)

        for raw_feat in raw_features:
            buffer.append(raw_feat)
            eng_feat = self.feature_engineer.engineer_features(raw_feat, buffer)
            engineered_features.append(eng_feat)

        engineered_features = np.array(engineered_features)

        # Normalize
        if self.feature_engineer.feature_mean is not None:
            engineered_features = self.feature_engineer.normalize(engineered_features)

        return torch.FloatTensor(engineered_features), torch.FloatTensor(targets), seq_ix


# ============================================================================
# 5. MODEL ARCHITECTURE
# ============================================================================

class LOBNetHybrid(nn.Module):
    """Hybrid CNN-LSTM-Attention model for LOB prediction"""

    def __init__(
        self,
        input_dim=48,
        cnn_channels=[48, 64, 64],
        cnn_kernel_size=3,
        lstm_hidden_dim=128,
        lstm_layers=2,
        lstm_dropout=0.2,
        attention_heads=4,
        fc_dims=[256, 128, 64, 2],
        fc_dropout=[0.3, 0.2, 0.0]
    ):
        super(LOBNetHybrid, self).__init__()

        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers

        # CNN Block
        self.conv1 = nn.Conv1d(input_dim, cnn_channels[1], kernel_size=cnn_kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_channels[1])
        self.conv2 = nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size=cnn_kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_channels[2])

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            cnn_channels[2],
            lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )

        # Multi-head attention
        lstm_output_dim = lstm_hidden_dim * 2
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=attention_heads,
            dropout=0.1,
            batch_first=True
        )

        # Projection head
        self.fc_layers = nn.ModuleList()
        prev_dim = fc_dims[0]
        for i, dim in enumerate(fc_dims[1:]):
            self.fc_layers.append(nn.Linear(prev_dim, dim))
            if i < len(fc_dropout) and fc_dropout[i] > 0:
                self.fc_layers.append(nn.Dropout(fc_dropout[i]))
            if i < len(fc_dims) - 2:
                self.fc_layers.append(nn.ReLU())
            prev_dim = dim

        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass"""
        batch_size, seq_len, _ = x.shape

        # CNN block
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)

        # LSTM block
        lstm_out, _ = self.lstm(x)

        # Attention block
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Residual connection
        x = lstm_out + attn_out

        # Projection head
        for layer in self.fc_layers:
            x = layer(x)

        return x


# ============================================================================
# 6. LOSS AND METRICS
# ============================================================================

class WeightedPearsonLoss(nn.Module):
    """Weighted Pearson Correlation Loss - optimizes competition metric"""

    def __init__(self):
        super(WeightedPearsonLoss, self).__init__()

    def forward(self, predictions, targets):
        predictions = torch.clamp(predictions, -6, 6)
        weights = torch.abs(targets)

        corr_t0 = self.weighted_pearson(
            predictions[:, :, 0].flatten(),
            targets[:, :, 0].flatten(),
            weights[:, :, 0].flatten()
        )

        corr_t1 = self.weighted_pearson(
            predictions[:, :, 1].flatten(),
            targets[:, :, 1].flatten(),
            weights[:, :, 1].flatten()
        )

        return -0.5 * (corr_t0 + corr_t1)

    def weighted_pearson(self, pred, target, weight):
        pred_mean = (pred * weight).sum() / (weight.sum() + 1e-8)
        target_mean = (target * weight).sum() / (weight.sum() + 1e-8)

        pred_centered = pred - pred_mean
        target_centered = target - target_mean

        numerator = (weight * pred_centered * target_centered).sum()
        denominator = torch.sqrt(
            (weight * pred_centered**2).sum() *
            (weight * target_centered**2).sum()
        )

        return numerator / (denominator + 1e-8)


def weighted_pearson_correlation_np(predictions, targets, weights):
    """Numpy version for evaluation"""
    predictions = np.clip(predictions, -6, 6)

    pred_mean = np.sum(predictions * weights) / np.sum(weights)
    target_mean = np.sum(targets * weights) / np.sum(weights)

    pred_centered = predictions - pred_mean
    target_centered = targets - target_mean

    numerator = np.sum(weights * pred_centered * target_centered)
    denominator = np.sqrt(
        np.sum(weights * pred_centered**2) *
        np.sum(weights * target_centered**2)
    )

    return numerator / (denominator + 1e-8)


# ============================================================================
# 7. TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device='cpu'):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for features, targets, seq_ids in tqdm(dataloader, desc="Training", leave=False):
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions[:, 99:, :], targets[:, 99:, :])
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device='cpu'):
    """Evaluate model"""
    model.eval()

    all_predictions = []
    all_targets = []
    all_weights = []

    with torch.no_grad():
        for features, targets, seq_ids in tqdm(dataloader, desc="Evaluating", leave=False):
            features = features.to(device)
            predictions = model(features)

            pred_steps = predictions[:, 99:, :].cpu().numpy()
            target_steps = targets[:, 99:, :].numpy()
            weights = np.abs(target_steps)

            all_predictions.append(pred_steps)
            all_targets.append(target_steps)
            all_weights.append(weights)

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    weights = np.concatenate(all_weights, axis=0)

    score_t0 = weighted_pearson_correlation_np(
        predictions[:, :, 0].flatten(),
        targets[:, :, 0].flatten(),
        weights[:, :, 0].flatten()
    )

    score_t1 = weighted_pearson_correlation_np(
        predictions[:, :, 1].flatten(),
        targets[:, :, 1].flatten(),
        weights[:, :, 1].flatten()
    )

    final_score = (score_t0 + score_t1) / 2

    return {
        'score': final_score,
        'score_t0': score_t0,
        'score_t1': score_t1
    }


def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler,
                num_epochs, device, model_name):
    """Complete training loop"""
    best_score = -float('inf')
    patience_counter = 0
    patience = 10

    history = {
        'train_loss': [],
        'valid_score': [],
        'valid_score_t0': [],
        'valid_score_t1': []
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)

        # Validate
        scores = evaluate(model, valid_loader, device)
        history['valid_score'].append(scores['score'])
        history['valid_score_t0'].append(scores['score_t0'])
        history['valid_score_t1'].append(scores['score_t1'])

        # Update learning rate
        scheduler.step()

        print(f"  Loss: {train_loss:.6f} | Score: {scores['score']:.6f} (t0: {scores['score_t0']:.6f}, t1: {scores['score_t1']:.6f})")

        # Save best model
        if scores['score'] > best_score:
            best_score = scores['score']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'score': best_score,
                'history': history
            }, WORK_DIR / 'weights' / f'{model_name}.pt')
            print(f"  ✓ BEST! Saved with score: {best_score:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered")
                break

    return history, best_score


# ============================================================================
# 8. ONNX EXPORT
# ============================================================================

def export_to_onnx(model, model_name, device):
    """Export PyTorch model to ONNX"""
    model.eval()

    dummy_input = torch.randn(1, 1000, 48).to(device)
    onnx_path = WORK_DIR / 'weights' / f'{model_name}.onnx'

    print(f"\n  Exporting {model_name} to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence'},
            'output': {0: 'batch_size', 1: 'sequence'}
        }
    )

    # Verify ONNX model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    file_size = onnx_path.stat().st_size / (1024**2)
    print(f"  ✓ ONNX export complete: {file_size:.2f} MB")

    return onnx_path


# ============================================================================
# 9. SUBMISSION CREATION
# ============================================================================

def create_submission_package(model_name, feature_engineer):
    """Create submission ZIP for competition"""

    submission_dir = WORK_DIR / 'submissions' / model_name
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Save normalization stats
    norm_stats = {
        'mean': feature_engineer.feature_mean,
        'std': feature_engineer.feature_std
    }
    np.savez(submission_dir / 'normalization_stats.npz', **norm_stats)

    # Copy ONNX model
    shutil.copy(
        WORK_DIR / 'weights' / f'{model_name}.onnx',
        submission_dir / 'model.onnx'
    )

    # Create solution.py
    solution_code = '''"""
LOBNet-Hybrid Model for Wunder Predictorium
Combines CNN, BiLSTM, and Attention for LOB prediction
"""

import numpy as np
import onnxruntime as ort
from collections import deque
from typing import Optional


class PredictionModel:
    """Stateful prediction model for LOB time series"""

    def __init__(self):
        # Initialize ONNX session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            "model.onnx",
            session_options,
            providers=['CPUExecutionProvider']
        )

        # Load normalization statistics
        stats = np.load("normalization_stats.npz")
        self.feature_mean = stats['mean']
        self.feature_std = stats['std']

        # State management
        self.current_seq_ix = None
        self.feature_buffer = deque(maxlen=20)
        self.sequence_features = []

    def predict(self, data_point) -> Optional[np.ndarray]:
        """Generate prediction for a single timestep"""
        # Handle sequence reset
        if data_point.seq_ix != self.current_seq_ix:
            if len(self.sequence_features) > 0:
                self._process_sequence()
            self._reset_state()
            self.current_seq_ix = data_point.seq_ix

        # Engineer and normalize features
        raw_features = data_point.state
        self.feature_buffer.append(raw_features)
        engineered = self._engineer_features(raw_features)
        normalized = self._normalize(engineered)

        # Store features
        self.sequence_features.append(normalized)

        # Return prediction if needed
        if data_point.need_prediction:
            if len(self.sequence_features) == 1000:
                predictions = self._run_inference()
                return predictions[data_point.step_in_seq]
            else:
                # Sequence still building
                return np.zeros(2, dtype=np.float32)

        return None

    def _process_sequence(self):
        """Process accumulated sequence"""
        if len(self.sequence_features) > 0:
            self._run_inference()

    def _run_inference(self) -> np.ndarray:
        """Run ONNX inference on full sequence"""
        # Pad if needed
        while len(self.sequence_features) < 1000:
            self.sequence_features.append(np.zeros(48, dtype=np.float32))

        # Prepare input
        features_array = np.array(self.sequence_features[:1000], dtype=np.float32)
        features_array = features_array.reshape(1, 1000, 48)

        # Run inference
        outputs = self.session.run(None, {'input': features_array})
        predictions = outputs[0][0]

        # Clip to required range
        return np.clip(predictions, -6.0, 6.0)

    def _engineer_features(self, raw_features: np.ndarray) -> np.ndarray:
        """Create derived features from raw LOB state"""
        prices = raw_features[:12]
        volumes = raw_features[12:24]
        trade_prices = raw_features[24:28]
        trade_volumes = raw_features[28:32]

        # Price features
        mid_price = (prices[0] + prices[1]) / 2
        spread = prices[1] - prices[0]
        weighted_mid = (prices[0] * volumes[1] + prices[1] * volumes[0]) / (volumes[0] + volumes[1] + 1e-8)
        price_imbalance = (volumes[0] - volumes[1]) / (volumes[0] + volumes[1] + 1e-8)
        relative_spread = spread / (mid_price + 1e-8)
        price_depth_diff = prices[0] - prices[2]

        # Volume features
        total_volume = np.sum(volumes)
        volume_concentration = volumes[0] / (total_volume + 1e-8)
        volume_ratio = volumes[0] / (volumes[2] + 1e-8)
        log_volume = np.log1p(total_volume)

        # Trade features
        avg_trade_price = np.mean(trade_prices)
        avg_trade_volume = np.mean(trade_volumes)

        # Temporal features
        if len(self.feature_buffer) >= 5:
            recent_states = np.array(list(self.feature_buffer))
            recent_mid_prices = (recent_states[-5:, 0] + recent_states[-5:, 1]) / 2
            price_momentum = np.mean(np.diff(recent_mid_prices))
            volume_volatility = np.std(recent_states[-5:, 12])
        else:
            price_momentum = 0.0
            volume_volatility = 0.0

        # Combine features
        engineered = np.array([
            mid_price, spread, weighted_mid, price_imbalance,
            total_volume, volume_concentration, price_momentum, avg_trade_price,
            relative_spread, price_depth_diff, volume_ratio, log_volume,
            avg_trade_volume, np.std(volumes), volume_volatility,
            np.max(volumes) - np.min(volumes)
        ], dtype=np.float32)

        return np.concatenate([raw_features, engineered])

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Standardize features"""
        return (features - self.feature_mean) / self.feature_std

    def _reset_state(self):
        """Reset state for new sequence"""
        self.feature_buffer.clear()
        self.sequence_features = []
'''

    # Write solution.py
    with open(submission_dir / 'solution.py', 'w', encoding='utf-8') as f:
        f.write(solution_code)

    # Create submission ZIP
    submission_zip = WORK_DIR / f'submission_{model_name}.zip'

    with zipfile.ZipFile(submission_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(submission_dir / 'solution.py', arcname='solution.py')
        zipf.write(submission_dir / 'model.onnx', arcname='model.onnx')
        zipf.write(submission_dir / 'normalization_stats.npz', arcname='normalization_stats.npz')

    file_size = submission_zip.stat().st_size / (1024**2)
    print(f"\n  ✓ Submission package created: {submission_zip.name} ({file_size:.2f} MB)")

    return submission_zip


# ============================================================================
# 10. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  WUNDER LOB PREDICTORIUM - COMPLETE PIPELINE")
    print("  Training 3 model variants and creating submissions")
    print("="*70 + "\n")

    # ========================================================================
    # STEP 1: Extract and Load Data
    # ========================================================================

    print("STEP 1: Extracting and loading data...")
    train_path = DATA_DIR / "train.parquet"
    valid_path = DATA_DIR / "valid.parquet"

    if not train_path.exists():
        print("  Extracting starter pack...")
        with zipfile.ZipFile(STARTER_PACK_PATH, 'r') as zip_ref:
            zip_ref.extract('competition_package/datasets/train.parquet', WORK_DIR)
            zip_ref.extract('competition_package/datasets/valid.parquet', WORK_DIR)
            zip_ref.extract('competition_package/utils.py', WORK_DIR)

        # Move files to correct location
        import shutil
        shutil.move(WORK_DIR / 'competition_package' / 'datasets' / 'train.parquet', train_path)
        shutil.move(WORK_DIR / 'competition_package' / 'datasets' / 'valid.parquet', valid_path)
        shutil.move(WORK_DIR / 'competition_package' / 'utils.py', WORK_DIR / 'utils.py')

        # Clean up
        shutil.rmtree(WORK_DIR / 'competition_package', ignore_errors=True)
        shutil.rmtree(WORK_DIR / '__MACOSX', ignore_errors=True)

        print("  ✓ Extraction complete")
    else:
        print("  ✓ Data already extracted")

    print("\n  Loading parquet files...")
    train_df = pd.read_parquet(train_path)
    valid_df = pd.read_parquet(valid_path)
    print(f"  ✓ Train: {train_df['seq_ix'].nunique():,} sequences")
    print(f"  ✓ Valid: {valid_df['seq_ix'].nunique():,} sequences")

    # ========================================================================
    # STEP 2: Feature Engineering
    # ========================================================================

    print("\n" + "="*70)
    print("STEP 2: Feature engineering...")
    feature_engineer = FeatureEngineer()
    feature_engineer.fit_normalization(train_df)

    # ========================================================================
    # STEP 3: Create Datasets
    # ========================================================================

    print("\n" + "="*70)
    print("STEP 3: Creating datasets...")

    # Use subset for faster training (remove [:500] for full training)
    train_sequences = train_df['seq_ix'].unique()[:500]
    train_subset = train_df[train_df['seq_ix'].isin(train_sequences)]

    train_dataset = LOBDataset(train_subset, feature_engineer, mode='train')
    valid_dataset = LOBDataset(valid_df, feature_engineer, mode='valid')

    # ========================================================================
    # STEP 4: Train Multiple Models
    # ========================================================================

    print("\n" + "="*70)
    print("STEP 4: Training models...")
    print("  Training 3 model variants with different hyperparameters")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Using device: {device}")

    results = {}

    for config_name, config in MODEL_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"  Training: {config['name']}")
        print(f"  Config: {config_name} | LSTM: {config['lstm_hidden_dim']} | Layers: {config['lstm_layers']} | Heads: {config['attention_heads']}")
        print(f"{'='*70}")

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
        valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=0)

        # Initialize model
        model = LOBNetHybrid(
            lstm_hidden_dim=config['lstm_hidden_dim'],
            lstm_layers=config['lstm_layers'],
            attention_heads=config['attention_heads']
        ).to(device)

        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Training setup
        criterion = WeightedPearsonLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )

        # Train
        history, best_score = train_model(
            model, train_loader, valid_loader, criterion, optimizer, scheduler,
            num_epochs=config['epochs'], device=device, model_name=config['name']
        )

        results[config_name] = {
            'score': best_score,
            'history': history
        }

        print(f"\n  ✓ {config['name']} complete! Best score: {best_score:.6f}")

        # Load best model for ONNX export
        checkpoint = torch.load(WORK_DIR / 'weights' / f"{config['name']}.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Export to ONNX
        export_to_onnx(model, config['name'], device)

        # Create submission package
        create_submission_package(config['name'], feature_engineer)

    # ========================================================================
    # STEP 5: Summary
    # ========================================================================

    print("\n" + "="*70)
    print("  TRAINING COMPLETE!")
    print("="*70)

    print("\nResults Summary:")
    for config_name, result in results.items():
        config = MODEL_CONFIGS[config_name]
        print(f"  {config['name']:20s} | Score: {result['score']:.6f}")

    print("\n" + "="*70)
    print("  SUBMISSION PACKAGES CREATED:")
    print("="*70)

    for config_name, config in MODEL_CONFIGS.items():
        submission_file = WORK_DIR / f"submission_{config['name']}.zip"
        if submission_file.exists():
            size = submission_file.stat().st_size / (1024**2)
            print(f"  ✓ {submission_file.name} ({size:.2f} MB)")

    print("\n" + "="*70)
    print("  ALL DONE! Ready to submit to competition platform")
    print("="*70 + "\n")
