#!/usr/bin/env python3
"""
prep_msft_features.py

Prep features for Microsoft earnings-phrase prediction.

Input parquet shape (one row per speaker turn / block):
  columns: ['speaker','text','word_count','ticker','quarter']
  - 'quarter' like "Q4 2025" (any ticker allowed; script is generic)

Output parquet shape (one row per ticker+quarter):
  ['ticker','quarter','total_word_count','num_speakers','avg_words_per_speaker',
   <15 binary phrase cols>,
   <prior features per phrase: prev_hit_1q, prev_hit_rate_4q, prev_hit_count_4q>]
"""

import argparse
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------- 1) Phrase dictionary (extend as you like) ----------
PHRASES: Dict[str, List[str]] = {
    "LinkedIn": ["linkedin"],
    "Copilot": ["copilot"],
    "M365": ["m365"],
    "Windows": ["windows"],
    "Azure": ["azure"],
    "Activision": ["activision"],
    "Phi": ["phi"],
    "Cybersecurity": ["cybersecurity"],
    "Stargate": ["stargate"],
    "Teams": ["teams"],
    "CapEx": ["capex"],
    "AI": ["ai"],
    "Cloud": ["cloud"],
    "OpenAI": ["openai"],
    "Gaming": ["gaming"],
}


# ---------- 2) Utilities ----------
Q_PAT = re.compile(r"Q\s*([1-4])\s*([12]\d{3})")  # matches "Q4 2025", "Q1 2023", etc.


def parse_quarter_key(qstr: str) -> Tuple[int, int]:
    """Parse "Q4 2025" -> (2025, 4)."""
    if not isinstance(qstr, str):
        return (0, 0)
    m = Q_PAT.search(qstr)
    if not m:
        return (0, 0)
    q = int(m.group(1))
    y = int(m.group(2))
    return (y, q)


# ---------- 2b) Strict regex builder ----------
def build_phrase_regex(variants: List[str]) -> re.Pattern:
    """
    Build a case-insensitive regex that matches only *exact* word occurrences,
    respecting word boundaries (\b) and disallowing substrings.
    Example: 'ai' will match 'AI' or 'ai,' but not 'said' or 'OpenAI'.
    """
    patterns = []
    for v in variants:
        v = v.strip().lower()
        # For multi-word phrases, we allow flexible whitespace (e.g. "capital expenditure")
        v_escaped = re.escape(v).replace(r"\ ", r"\s+")
        # Add word boundaries around the entire phrase
        patterns.append(rf"\b{v_escaped}\b")
    pattern = "(" + "|".join(patterns) + ")"
    return re.compile(pattern, flags=re.IGNORECASE)


PHRASE_REGEX = {k: build_phrase_regex(vs) for k, vs in PHRASES.items()}


def detect_phrases_in_text(text: str) -> Dict[str, int]:
    """Return {phrase: 0/1} for whether each phrase appears in the given text."""
    text = text or ""
    out = {}
    for name, pat in PHRASE_REGEX.items():
        out[name] = 1 if pat.search(text) else 0
    return out


# ---------- 3) Main prep pipeline ----------
def prepare_features(input_parquet: str, output_parquet: str) -> None:
    # Load
    df = pd.read_parquet(input_parquet)

    # Basic sanity
    required_cols = {"speaker", "text", "word_count", "ticker", "quarter"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure dtypes
    df["speaker"] = df["speaker"].astype("string")
    df["text"] = df["text"].astype("string")
    df["ticker"] = df["ticker"].astype("string")
    df["quarter"] = df["quarter"].astype("string")
    df["word_count"] = pd.to_numeric(df["word_count"], errors="coerce").fillna(0).astype(int)

    # Phrase flags per row
    phrase_df = df["text"].apply(lambda t: pd.Series(detect_phrases_in_text(str(t))))
    df = pd.concat([df, phrase_df], axis=1)

    # Aggregate to quarter-level
    phrase_cols = list(PHRASES.keys())
    agg_dict = {c: "max" for c in phrase_cols}
    agg_dict.update({"word_count": "sum"})
    quarterly = (
        df.groupby(["ticker", "quarter"], as_index=False)
          .agg(agg_dict)
          .rename(columns={"word_count": "total_word_count"})
    )

    # Add simple structural features
    spk = (
        df.groupby(["ticker", "quarter"])["speaker"]
          .nunique()
          .reset_index()
          .rename(columns={"speaker": "num_speakers"})
    )
    quarterly = quarterly.merge(spk, on=["ticker", "quarter"], how="left")
    quarterly["num_speakers"] = quarterly["num_speakers"].fillna(0).astype(int)
    quarterly["avg_words_per_speaker"] = (
        quarterly["total_word_count"] / quarterly["num_speakers"].replace(0, np.nan)
    ).fillna(0.0)

    # Sort by time using parsed quarter key (per ticker)
    quarterly[["year_key", "qtr_key"]] = quarterly["quarter"].apply(
        lambda s: pd.Series(parse_quarter_key(s))
    )
    quarterly = quarterly.sort_values(["ticker", "year_key", "qtr_key"]).reset_index(drop=True)

    # Prior-mention features per phrase (grouped by ticker)
    def add_prior_features(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        for col in phrase_cols:
            # Previous quarter hit (shifted)
            g[f"{col}_prev_hit_1q"] = g[col].shift(1)
            # Rolling 4-quarter stats (exclude current by shifting before rolling)
            prev = g[col].shift(1)
            g[f"{col}_prev_hit_rate_4q"] = prev.rolling(window=4, min_periods=1).mean()
            g[f"{col}_prev_hit_count_4q"] = prev.rolling(window=4, min_periods=1).sum()
        return g

    quarterly = (
        quarterly.groupby("ticker", group_keys=False)
                 .apply(add_prior_features)
                 .reset_index(drop=True)
    )

    # Fill NaNs in new numeric cols
    prior_cols = [c for c in quarterly.columns if c.endswith(("_prev_hit_1q", "_prev_hit_rate_4q", "_prev_hit_count_4q"))]
    quarterly[prior_cols] = quarterly[prior_cols].fillna(0.0)

    # Final tidy: drop helper sort keys
    quarterly = quarterly.drop(columns=["year_key", "qtr_key"])

    # Save
    quarterly.to_parquet(output_parquet, index=False)
    print(f"✅ Wrote features: {output_parquet}")
    print(f"Rows: {len(quarterly)}, Columns: {len(quarterly.columns)}")


# ---------- 4) CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Prep phrase-mention features from transcript parquet.")
    ap.add_argument("--in", dest="input_parquet", required=True, help="Path to input parquet with per-turn transcripts")
    ap.add_argument("--out", dest="output_parquet", required=True, help="Path to output features parquet")
    args = ap.parse_args()
    prepare_features(args.input_parquet, args.output_parquet)


if __name__ == "__main__":
    main()

    """
    Run command line:   
    python3 feature_prep.py \
  --in /Users/zachburns/Desktop/talk_tell/Data/combined_transcripts.parquet \
#   --out /Users/zachburns/Desktop/talk_tell/Data/msft_features.parquet
    """