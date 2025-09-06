# combine_datasets.py
import argparse
import os
from pathlib import Path

import pandas as pd


def load_wdw_passport(path: str) -> pd.DataFrame:
    """Load WDW Passport daily crowd index."""
    df = pd.read_csv(path, parse_dates=["date"])
    # Ensure expected columns
    if "crowd_index" not in df.columns:
        raise ValueError(
            f"{path} must have a 'crowd_index' column (int 1–10) and a 'date' column."
        )
    # Standardize types
    df["crowd_index"] = pd.to_numeric(df["crowd_index"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df[["date", "crowd_index"]]


def aggregate_queue_times(path: str, level: str) -> pd.DataFrame:
    """
    Aggregate Queue-Times ride-level CSV to daily metrics.

    level options:
      - overall: one row per date (all rides/parks together)
      - park:    one row per date x park_id
      - ride:    one row per date x ride
    """
    q = pd.read_csv(path, parse_dates=["timestamp_utc"])

    # Basic sanity checks
    required_cols = {"timestamp_utc", "wait_time", "is_open"}
    if not required_cols.issubset(set(q.columns)):
        raise ValueError(
            f"{path} must include columns: {sorted(required_cols)}; found {sorted(q.columns)}"
        )

    # Normalize types
    q["wait_time"] = pd.to_numeric(q["wait_time"], errors="coerce")
    # Some sources store is_open as True/False, some as 1/0 strings
    q["is_open"] = (
        q["is_open"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
    )

    # Derive date
    # (timestamp_utc may be naive; we only need the calendar date)
    q["date"] = q["timestamp_utc"].dt.tz_localize(None).dt.date
    q["date"] = pd.to_datetime(q["date"])  # back to Timestamp for merging

    # Grouping keys by level
    if level == "overall":
        keys = ["date"]
    elif level == "park":
        if "park_id" not in q.columns:
            raise ValueError("Queue-Times file is missing 'park_id' for level='park'.")
        keys = ["date", "park_id"]
    elif level == "ride":
        if "ride" not in q.columns:
            raise ValueError("Queue-Times file is missing 'ride' for level='ride'.")
        keys = ["date", "ride"]
    else:
        raise ValueError("level must be one of: overall, park, ride")

    agg = (
        q.groupby(keys, dropna=False)
        .agg(
            avg_wait_queue=("wait_time", "mean"),
            med_wait_queue=("wait_time", "median"),
            p95_wait_queue=("wait_time", lambda s: s.quantile(0.95)),
            pct_open=("is_open", "mean"),  # fraction of samples where the ride was open
            samples=("wait_time", "count"),
            unique_rides=("ride", "nunique"),
        )
        .reset_index()
    )

    # pct_open -> percentage 0..100
    agg["pct_open"] = (agg["pct_open"] * 100).round(2)
    # Round waits a bit for readability
    for c in ["avg_wait_queue", "med_wait_queue", "p95_wait_queue"]:
        agg[c] = agg[c].round(2)

    return agg


def main():
    ap = argparse.ArgumentParser(
        description="Combine WDW Passport (crowd index) with optional Queue-Times aggregates."
    )
    ap.add_argument("--wdw", required=True, help="CSV from scrape_wdwpassport.py")
    ap.add_argument(
        "--queue",
        required=False,
        help="Optional CSV from collect_queue_times.py (ride-level; can be large)",
    )
    ap.add_argument(
        "--queue_level",
        default="overall",
        choices=["overall", "park", "ride"],
        help="Aggregation level for Queue-Times (default: overall).",
    )
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    # Ensure output dir exists
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Load WDW Passport
    wdw = load_wdw_passport(args.wdw)

    # If no queue file, just save WDW Passport as-is
    if not args.queue:
        wdw.sort_values("date").to_csv(args.out, index=False)
        print(f"Wrote {args.out} with {len(wdw)} rows (WDW Passport only).")
        return

    # Aggregate Queue-Times and merge on date
    q_agg = aggregate_queue_times(args.queue, args.queue_level)

    # Decide merge keys
    if args.queue_level == "overall":
        on = ["date"]
        combined = pd.merge(wdw, q_agg, on=on, how="outer")
    elif args.queue_level == "park":
        # Outer join on date; crowd_index is resort-wide,
        # so it will repeat for each park_id that day.
        on = ["date"]
        combined = pd.merge(q_agg, wdw, on=on, how="left")
    else:  # ride level
        on = ["date"]
        combined = pd.merge(q_agg, wdw, on=on, how="left")

    combined = combined.sort_values(on)

    combined.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(combined)} rows.")


if __name__ == "__main__":
    main()
