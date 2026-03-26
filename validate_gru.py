"""Validate GRU model step-by-step using ScorerStepByStep"""
import sys
import time
import importlib.util

# Load solution_gru.py as the solution module
spec = importlib.util.spec_from_file_location("solution", r"D:\Wunder Fund\Claude\solution_gru.py")
solution_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution_module)

# Patch sys.modules so utils can find it
sys.modules['solution'] = solution_module

from utils import ScorerStepByStep
import numpy as np

print("Loading scorer...", flush=True)
scorer = ScorerStepByStep(r"D:\Wunder Fund\Claude\datasets\valid.parquet")
print(f"Loaded: {len(scorer.dataset)} rows", flush=True)

print("Creating model...", flush=True)
model = solution_module.PredictionModel()

print("Running step-by-step scoring...", flush=True)
t0 = time.time()
results = scorer.score(model)
elapsed = time.time() - t0

print(f"\n===== RESULTS =====", flush=True)
print(f"Keys: {list(results.keys())}", flush=True)
for k, v in results.items():
    print(f"  {k}: {v:.6f}", flush=True)
print(f"\nTime: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
n_seqs = len(scorer.dataset['seq_ix'].unique())
print(f"Sequences: {n_seqs}", flush=True)
time_per_seq = elapsed / n_seqs
print(f"Time/seq: {time_per_seq:.2f}s", flush=True)
print(f"Est. for 1500 test seqs: {time_per_seq * 1500 / 60:.1f} min", flush=True)
