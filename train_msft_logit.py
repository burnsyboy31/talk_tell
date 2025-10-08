#!/usr/bin/env python3
"""
train_msft_logit.py

Calibrated logistic regression for multi-label phrase mention prediction (Microsoft only).
- Uses prior features only (prev_hit_1q, prev_hit_rate_4q, prev_hit_count_4q) to avoid leakage.
- Leave-One-Out CV to get out-of-sample probabilities on tiny datasets.
- Compares to current Kalshi "Yes" prices (implied probabilities).

Usage:
  python3 train_msft_logit.py --in <features.parquet> --out <oos_results.csv>
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import brier_score_loss

PHRASES = [
    "LinkedIn","Copilot","M365","Windows","Azure",
    "Activision","Phi","Cybersecurity","Stargate","Teams",
    "CapEx","AI","Cloud","OpenAI","Gaming"
]

# Fill/update these with your latest Yes-prices in cents (None if unknown)
KALSHI_YES_CENTS = {
    "LinkedIn":98, "Copilot":99, "M365":95, "Windows":99, "Azure":99,
    "Activision":70, "Phi":50, "Cybersecurity":40, "Stargate":25, "Teams":98,
    "CapEx":91, "AI":99, "Cloud":99, "OpenAI":91, "Gaming":None
}
KALSHI_IMP = {k:(v/100.0 if v is not None else np.nan) for k,v in KALSHI_YES_CENTS.items()}

def prior_cols_for(phrase: str):
    return [f"{phrase}_prev_hit_1q", f"{phrase}_prev_hit_rate_4q", f"{phrase}_prev_hit_count_4q"]

def loo_oos_probs(X: np.ndarray, y: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Leave-One-Out OOS probabilities with robust calibration fallback.
    - If a fold has both classes but too few samples for 3-fold CV, drop to cv=2.
    - If any class has <2 samples, skip calibration and use plain logistic.
    - If the fold is single-class, use Laplace-smoothed prior.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import LeaveOneOut
    import numpy as np

    n = len(y)
    oos = np.zeros(n, dtype=float)
    loo = LeaveOneOut()

    for train_idx, test_idx in loo.split(X):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr = y[train_idx]

        # Single-class fold → smoothed prior
        if len(np.unique(ytr)) == 1:
            p = (ytr.mean() * len(ytr) + 1) / (len(ytr) + 2)  # Laplace smoothing
            oos[test_idx[0]] = p
            continue

        # Count per class in the training fold
        _, counts = np.unique(ytr, return_counts=True)
        min_class = counts.min()

        base = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed)

        try:
            if min_class >= 3:
                # ok to use 3-fold Platt
                clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
                clf.fit(Xtr, ytr)
                oos[test_idx[0]] = clf.predict_proba(Xte)[:, 1][0]
            elif min_class >= 2:
                # not enough for 3-fold; use 2-fold Platt
                clf = CalibratedClassifierCV(base, method="sigmoid", cv=2)
                clf.fit(Xtr, ytr)
                oos[test_idx[0]] = clf.predict_proba(Xte)[:, 1][0]
            else:
                # too few per class to calibrate → plain logistic
                base.fit(Xtr, ytr)
                oos[test_idx[0]] = base.predict_proba(Xte)[:, 1][0]
        except Exception:
            # any unexpected calibration issue → plain logistic as safe fallback
            base.fit(Xtr, ytr)
            oos[test_idx[0]] = base.predict_proba(Xte)[:, 1][0]

    return oos

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Path to features parquet (from prep script)")
    ap.add_argument("--out", dest="out_path", required=False, help="Optional: path to write OOS results CSV")
    args = ap.parse_args()

    # Load features
    df = pd.read_parquet(args.in_path).copy()
    # Ensure sorted by time (your 'quarter' strings like 'Q4 2025' will already be ordered by prep script, but keep this)
    df = df.reset_index(drop=True)

    rows = []
    for phrase in PHRASES:
        if phrase not in df.columns:
            print(f"[WARN] Missing label column for phrase: {phrase}. Skipping.")
            continue

        # y label and X = prior features only
        y = df[phrase].astype(int).values
        X = df[prior_cols_for(phrase)].astype(float).values

        # Edge case: if all labels are identical across quarters, use smoothed prior for all rows
        if y.sum() == 0 or y.sum() == len(y):
            p_all = (y.mean()*len(y) + 1) / (len(y) + 2)
            oos = np.full(len(y), p_all, dtype=float)
        else:
            oos = loo_oos_probs(X, y)

        for i in range(len(y)):
            rows.append({
                "quarter": df.loc[i, "quarter"],
                "phrase": phrase,
                "y_true": int(y[i]),
                "p_oos": float(oos[i]),
                "kalshi_p": KALSHI_IMP.get(phrase, np.nan),
                "edge": float(oos[i] - KALSHI_IMP.get(phrase, np.nan)) if np.isfinite(KALSHI_IMP.get(phrase, np.nan)) else np.nan
            })

    res = pd.DataFrame(rows).sort_values(["phrase", "quarter"]).reset_index(drop=True)

    # Print summary for latest quarter
    latest_q = df["quarter"].iloc[-1]
    latest = res[res["quarter"] == latest_q].copy()
    print(f"\n=== Latest quarter: {latest_q} ===")
    print(latest[["phrase", "p_oos", "kalshi_p", "edge"]]
          .sort_values("edge", ascending=False)
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Brier across all folds (where p_oos exists)
    valid = res.dropna(subset=["p_oos"])
    brier = brier_score_loss(valid["y_true"], valid["p_oos"])
    print(f"\nBrier score (all phrases, LOO OOS): {brier:.3f}")

    # Optional: write full OOS table
    if args.out_path:
        res.to_csv(args.out_path, index=False)
        print(f"\nSaved OOS results to {args.out_path}")

if __name__ == "__main__":
    main()

    
    """
python3 train_msft_logit.py \
  --in /Users/zachburns/Desktop/talk_tell/Data/msft_features.parquet \
  --out /Users/zachburns/Desktop/talk_tell/Data/msft_logit_oos.csv
  """