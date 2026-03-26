"""Validate ensemble solution step-by-step."""
import sys, time
sys.path.insert(0, r"D:\Wunder Fund\Claude")
from solution_ensemble import PredictionModel
from utils import ScorerStepByStep

model = PredictionModel()
scorer = ScorerStepByStep(r"D:\Wunder Fund\Claude\datasets\valid.parquet")
t0 = time.time()
result = scorer.score(model)
elapsed = time.time() - t0
print(f"\nResult: {result}")
print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
