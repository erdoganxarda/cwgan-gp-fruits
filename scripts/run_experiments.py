"""
Run the full data-size experiment grid.

Grid:
  sizes     = [100, 200, 400, 800, 1300]  (per class)
  scenarios = [real, synth, both]

Outputs all results to runs/clf/ as JSON files, plus a combined summary.

Usage:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --sizes 200 400 800
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from train_classifier import run


DEFAULT_SIZES = [100, 200, 400, 800, 1300]
SCENARIOS = ["real", "synth", "both"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--synth_dir", type=str, default="data_synth")
    parser.add_argument("--out_dir", type=str, default="runs/clf")
    args = parser.parse_args()

    cfg = Config()
    all_results = []

    for n in args.sizes:
        for scenario in SCENARIOS:
            print(f"\n{'='*60}")
            print(f"  Experiment: scenario={scenario}  n_per_class={n}")
            print(f"{'='*60}")
            result = run(cfg, scenario, n, args.synth_dir, args.out_dir)
            all_results.append(result)

    # save combined summary
    out_path = Path(args.out_dir)
    with open(out_path / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # print summary table
    print(f"\n{'='*70}")
    print(f"{'Size':>6} | {'Real':>8} | {'Synth':>8} | {'Both':>8} | {'Real(s)':>8} | {'Synth(s)':>8} | {'Both(s)':>8}")
    print(f"{'-'*70}")

    by_size = {}
    for r in all_results:
        key = r["n_per_class"]
        by_size.setdefault(key, {})[r["scenario"]] = r

    for n in args.sizes:
        row = by_size.get(n, {})
        accs = [f"{row.get(s, {}).get('test_accuracy', 0):.4f}" if s in row else "  N/A " for s in SCENARIOS]
        times = [f"{row.get(s, {}).get('train_time_sec', 0):>7.1f}" if s in row else "  N/A " for s in SCENARIOS]
        print(f"{n:>6} | {accs[0]:>8} | {accs[1]:>8} | {accs[2]:>8} | {times[0]:>8} | {times[1]:>8} | {times[2]:>8}")

    print(f"{'='*70}")
    print(f"Results saved to {out_path / 'all_results.json'}")


if __name__ == "__main__":
    main()
