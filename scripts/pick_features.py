"""Pick top-N features from shap_importance.json via cumulative coverage.

Reads model/models/{SYMBOL}_wf_{TAG}/shap_importance.json and writes
selected_features.json (same shape as the old trim120) with the rule:
  N = smallest N such that cum_share >= --cum-threshold, clipped to [--min-n, --max-n].

Usage:
    python -m scripts.pick_features --symbol BTC --tag sym278
    python -m scripts.pick_features --symbol BTC --tag sym278 --cum-threshold 0.95
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cum-threshold", type=float, default=0.95)
    ap.add_argument("--min-n", type=int, default=80)
    ap.add_argument("--max-n", type=int, default=150)
    args = ap.parse_args()

    model_dir = PROJECT_ROOT / "model" / "models" / f"{args.symbol}_wf_{args.tag}"
    shap_path = model_dir / "shap_importance.json"
    with open(shap_path) as f:
        shap = json.load(f)

    feats = shap["features"]  # sorted desc by mean_abs_shap
    n_total = len(feats)

    n_pick = None
    for i, row in enumerate(feats, start=1):
        if row["cum_share"] >= args.cum_threshold:
            n_pick = i
            break
    if n_pick is None:
        n_pick = n_total
    n_pick = max(args.min_n, min(args.max_n, n_pick))
    picked = feats[:n_pick]
    cum = picked[-1]["cum_share"]

    out = {
        "symbol": args.symbol,
        "source_tag": args.tag,
        "rule": f"cum_share >= {args.cum_threshold}, clipped to [{args.min_n}, {args.max_n}]",
        "n_selected": n_pick,
        "n_total": n_total,
        "cum_share": cum,
        "features": [r["feature"] for r in picked],
    }
    out_path = model_dir / "selected_features.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"picked {n_pick}/{n_total} features (cum_share={cum*100:.2f}%)")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
