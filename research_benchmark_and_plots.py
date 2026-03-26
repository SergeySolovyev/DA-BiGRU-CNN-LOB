import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from utils import ScorerStepByStep


ROOT = Path(r"D:\Wunder Fund\Claude")
DATASET_PATH = ROOT / "datasets" / "valid.parquet"
OUT_DIR = ROOT / "research_outputs"
OUT_DIR.mkdir(exist_ok=True)


def load_module(file_path: Path):
    spec = importlib.util.spec_from_file_location(file_path.stem, str(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_subset_parquet(source_path: Path, max_seqs: int) -> Path:
    df = pd.read_parquet(source_path)
    seqs = df["seq_ix"].unique()[:max_seqs]
    subset = df[df["seq_ix"].isin(seqs)].copy()
    subset_path = OUT_DIR / f"valid_subset_{max_seqs}_seqs.parquet"
    subset.to_parquet(subset_path, index=False)
    return subset_path


def run_benchmark(solution_file: Path, dataset_path: Path):
    result = {
        "solution": solution_file.name,
        "status": "ok",
        "t0": np.nan,
        "t1": np.nan,
        "weighted_pearson": np.nan,
        "runtime_sec": np.nan,
    }
    try:
        module = load_module(solution_file)
        sys.modules["solution"] = module
        model = module.PredictionModel()
        scorer = ScorerStepByStep(str(dataset_path))
        t0 = time.time()
        scores = scorer.score(model)
        elapsed = time.time() - t0
        result.update(
            {
                "t0": float(scores.get("t0", np.nan)),
                "t1": float(scores.get("t1", np.nan)),
                "weighted_pearson": float(scores.get("weighted_pearson", np.nan)),
                "runtime_sec": float(elapsed),
            }
        )
    except Exception as ex:
        result["status"] = f"error: {type(ex).__name__}: {ex}"
    return result


def plot_results(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return

    ok["efficiency"] = ok["weighted_pearson"] / ok["runtime_sec"]

    plt.figure(figsize=(10, 5))
    plt.bar(ok["solution"], ok["weighted_pearson"], color="#2E86DE")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Weighted Pearson")
    plt.title("Model quality on validation subset")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "chart_score.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(ok["solution"], ok["runtime_sec"], color="#E67E22")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Runtime (sec)")
    plt.title("Inference runtime on validation subset")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "chart_runtime.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(ok["solution"], ok["efficiency"], color="#27AE60")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Score / second")
    plt.title("Efficiency (quality per second)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "chart_efficiency.png", dpi=160)
    plt.close()


def main():
    max_seqs = int(os.getenv("RESEARCH_MAX_SEQS", "200"))

    default_solutions = [
        "solution_gru.py",
        "solution_ensemble.py",
        "solution_ensemble_v1v2.py",
        "solution_dual_bigru_cnn.py",
        "solution_ensemble_3model.py",
    ]
    selected = os.getenv("RESEARCH_SOLUTIONS", "").strip()
    if selected:
        solution_names = [x.strip() for x in selected.split(",") if x.strip()]
    else:
        solution_names = default_solutions
    solutions = [ROOT / name for name in solution_names]

    subset_path = build_subset_parquet(DATASET_PATH, max_seqs=max_seqs)
    print(f"Subset created: {subset_path}")

    rows = []
    for s in solutions:
        if not s.exists():
            rows.append(
                {
                    "solution": s.name,
                    "status": "missing_file",
                    "t0": np.nan,
                    "t1": np.nan,
                    "weighted_pearson": np.nan,
                    "runtime_sec": np.nan,
                }
            )
            continue
        print(f"Running: {s.name}")
        rows.append(run_benchmark(s, subset_path))

    result_df = pd.DataFrame(rows)
    result_df["runtime_min"] = result_df["runtime_sec"] / 60.0
    out_csv = OUT_DIR / "benchmark_results.csv"
    result_df.to_csv(out_csv, index=False)

    summary = {
        "dataset": str(subset_path),
        "max_seqs": max_seqs,
        "generated_files": [
            str(out_csv),
            str(OUT_DIR / "chart_score.png"),
            str(OUT_DIR / "chart_runtime.png"),
            str(OUT_DIR / "chart_efficiency.png"),
        ],
        "rows": result_df.to_dict(orient="records"),
    }
    with open(OUT_DIR / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    try:
        plot_results(result_df)
    except Exception as ex:
        print(f"Plot generation failed: {ex}")

    print("\n=== RESULTS ===")
    print(result_df.to_string(index=False))
    print(f"\nSaved to: {out_csv}")


if __name__ == "__main__":
    main()
