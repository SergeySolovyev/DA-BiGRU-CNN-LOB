"""Validate LightGBM solution step-by-step using ScorerStepByStep."""
import sys
import time
import importlib.util

# Load solution module
spec = importlib.util.spec_from_file_location("solution", r"D:\Wunder Fund\Claude\solution_lgbm.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PredictionModel = mod.PredictionModel

# Load utils
sys.path.insert(0, r"D:\Wunder Fund\Claude")
from utils import ScorerStepByStep

VALID_PATH = r"D:\Wunder Fund\Claude\datasets\valid.parquet"

print("Creating model...", flush=True)
model = PredictionModel()
print(f"  Model loaded OK", flush=True)

print(f"\nRunning step-by-step validation on {VALID_PATH}...", flush=True)
scorer = ScorerStepByStep(VALID_PATH)

t0 = time.time()
result = scorer.score(model)
elapsed = time.time() - t0

print(f"\n{'='*50}", flush=True)
print(f"Step-by-step validation results:", flush=True)
print(f"  t0: {result['t0']:.6f}", flush=True)
print(f"  t1: {result['t1']:.6f}", flush=True)
print(f"  weighted_pearson: {result['weighted_pearson']:.6f}", flush=True)
print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
print(f"  Sequences: {result.get('n_sequences', 'N/A')}", flush=True)
print(f"{'='*50}", flush=True)
