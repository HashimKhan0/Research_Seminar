#!/usr/bin/env python3
"""
data_loading.py

Minute‑level ETF data → three chronological splits
 • Train : 1 Jan 2023 – 30 Dec 2023
 • Valid : 1 Jan 2024 – 30 Mar 2024
 • Test  : 1 Apr 2024 – 1 Jan 2025
Log returns on the `close` column are computed before saving.
"""

import pandas as pd
from pathlib import Path
import numpy as np

# ─── CONFIG ────────────────────────────────────────────────────────────────
INPUT_FILE = "data.csv"   # raw minute bars
TRAIN_FILE = "train.csv"
TEST_FILE  = "test.csv"
VALID_FILE = "valid.csv"

TRAIN_START, TRAIN_END = "2023-01-01", "2023-12-30"
VALID_START, VALID_END = "2024-01-01", "2024-03-30"
TEST_START , TEST_END  = "2024-04-01", "2025-01-01"
# ────────────────────────────────────────────────────────────────────────────

def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add log returns column and drop the first nan row."""
    df = df.copy()
    df["log_price"] = np.log(df["close"])
    df["return"]    = df["log_price"].diff()
    return df.dropna()


def time_split() -> None:
    print(f"📁 Loading {INPUT_FILE} …")
    df = pd.read_csv(INPUT_FILE)
    # Drop symbol column if present
    if "symbol" in df.columns:
        df = df.drop(columns=["symbol"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("🔍 Computing log‑returns …")
    df = compute_log_returns(df)

    m_train = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)
    m_valid = (df["timestamp"] >= VALID_START) & (df["timestamp"] <= VALID_END)
    m_test  = (df["timestamp"] >= TEST_START)  & (df["timestamp"] <= TEST_END)

    train_df, valid_df, test_df = df[m_train], df[m_valid], df[m_test]
   
    print(f"✅ Train rows : {len(train_df):,}")
    print(f"✅ Valid rows : {len(valid_df):,}")
    print(f"✅ Test  rows : {len(test_df):,}")
    
    train_df.to_csv(TRAIN_FILE, index=False)
    valid_df.to_csv(VALID_FILE, index=False)
    test_df.to_csv(TEST_FILE , index=False)


    print(f"📄 Saved → {TRAIN_FILE}, {VALID_FILE}, {TEST_FILE}")


def main() -> None:
    # avoid accidental overwrite
    if all(Path(f).exists() for f in (TRAIN_FILE, VALID_FILE, TEST_FILE)):
        print("⚠️  Output files exist – delete them if you want to regenerate.")
        return
    time_split()


if __name__ == "__main__":
    main()