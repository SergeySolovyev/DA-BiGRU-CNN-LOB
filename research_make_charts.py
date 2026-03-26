import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path(r"D:\Wunder Fund\Claude\research_outputs")


def main():
    files = [
        OUT_DIR / "single_solution_gru.json",
        OUT_DIR / "single_solution_ensemble.json",
        OUT_DIR / "single_solution_ensemble_v1v2.json",
    ]
    rows = []
    for file in files:
        if file.exists():
            with open(file, "r", encoding="utf-8") as f:
                rows.append(json.load(f))

    if not rows:
        raise SystemExit("No single_solution_*.json files found")

    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"].copy()
    ok["efficiency"] = ok["weighted_pearson"] / ok["runtime_sec"]

    out_csv = OUT_DIR / "benchmark_results.csv"
    ok.to_csv(out_csv, index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(ok["solution"], ok["weighted_pearson"], color="#2E86DE")
    plt.title("Validation quality (Weighted Pearson)")
    plt.ylabel("score")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "chart_score.png", dpi=170)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(ok["solution"], ok["runtime_sec"], color="#E67E22")
    plt.title("Runtime on same validation subset")
    plt.ylabel("seconds")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "chart_runtime.png", dpi=170)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(ok["solution"], ok["efficiency"], color="#27AE60")
    plt.title("Efficiency = score / runtime")
    plt.ylabel("score per second")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "chart_efficiency.png", dpi=170)
    plt.close()

    with open(OUT_DIR / "benchmark_notes.md", "w", encoding="utf-8") as f:
        f.write("# Benchmark notes\n\n")
        f.write("Subset: 50 sequences from valid.parquet\n\n")
        f.write(ok[["solution", "t0", "t1", "weighted_pearson", "runtime_sec", "efficiency"]].to_csv(index=False))

    print(ok[["solution", "weighted_pearson", "runtime_sec", "efficiency"]].to_string(index=False))
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
