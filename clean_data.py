import argparse
import os
import re
import numpy as np
import pandas as pd

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clean_text(x: object) -> str | float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()

    # Remove repeated wrapping quotes (handles "Ride", 'Ride', """Ride""", ''Ride'')
    # Keep applying while both ends are quotes and there's content in the middle
    while len(s) >= 2 and ((s[0] == s[-1]) and s[0] in ("'", '"')):
        s = s[1:-1].strip()

    # Remove any stray leading/trailing quotes left over
    s = s.strip('"').strip("'").strip()

    # Collapse multiple spaces and weird whitespace
    s = re.sub(r"\s+", " ", s)

    return s if s else np.nan


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_wdw_passport(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    # Keep only expected columns if present
    keep = [c for c in df.columns if c.lower() in {"date", "crowd_index"}]
    if not keep or "date" not in [c.lower() for c in keep] or "crowd_index" not in [c.lower() for c in keep]:
        raise ValueError("WDW file must include 'date' and 'crowd_index' columns.")

    df = df[keep].rename(columns={c: c.lower() for c in keep})

    # Drop rows with missing date
    df = df.dropna(subset=["date"])

    # Coerce and bound crowd_index
    df["crowd_index"] = to_numeric(df["crowd_index"])
    df = df.dropna(subset=["crowd_index"])
    df = df[(df["crowd_index"] >= 1) & (df["crowd_index"] <= 10)]

    # De-duplicate by date (keep latest)
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    # Final sort & reset
    df = df.sort_values("date").reset_index(drop=True)
    return df

def clean_queue_times(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"timestamp_utc", "ride", "wait_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Queue-Times file missing required columns: {sorted(missing)}")

    # Parse timestamp; if any invalid -> NaT -> drop
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])

    # Normalize text columns (land, ride) and remove extra quotes
    if "land" not in df.columns:
        df["land"] = np.nan
    df["land"] = df["land"].apply(clean_text)
    df["ride"] = df["ride"].apply(clean_text)

    # Drop rows with missing ride after cleaning
    df = df.dropna(subset=["ride"])

    # Remove Trick-or-Treat overlay entries
    mask = df["ride"].str.contains("Trick-or-Treat Locations at Mickey's Not-So-Scary Halloween Party", case=False,
                                   na=False)
    df = df.loc[~mask]

    # Coerce numerics
    df["wait_time"] = to_numeric(df["wait_time"])
    df = df.dropna(subset=["wait_time"])

    # Bound wait_time to reasonable range (0..300 minutes)
    df = df[(df["wait_time"] >= 0) & (df["wait_time"] <= 300)]

    # Normalize is_open to boolean; missing -> False
    if "is_open" in df.columns:
        df["is_open"] = (
            df["is_open"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
            .fillna(False)
        )
    else:
        df["is_open"] = False

    # Park id to Int64 if present
    if "park_id" in df.columns:
        df["park_id"] = to_numeric(df["park_id"]).astype("Int64")

    # Add convenience columns
    df["timestamp_utc"] = df["timestamp_utc"].dt.tz_convert(None)  # keep UTC but naive for grouping if desired
    df["date"] = df["timestamp_utc"].dt.date
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["timestamp_utc"].dt.hour

    # Drop exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    return df

def main():
    ap = argparse.ArgumentParser(description="Clean WDW Passport and Queue-Times datasets (remove NAs, quotes, bounds).")
    ap.add_argument("--wdw", required=True, help="Path to WDW Passport CSV (date, crowd_index).")
    ap.add_argument("--queue", required=True, help="Path to Queue-Times CSV (ride-level snapshots).")
    ap.add_argument("--outdir", required=True, help="Directory to write cleaned CSVs.")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # WDW
    wdw_clean = clean_wdw_passport(args.wdw)
    wdw_out = os.path.join(args.outdir, "wdwpassport_clean.csv")
    wdw_clean.to_csv(wdw_out, index=False)
    print(f"Wrote {wdw_out} ({len(wdw_clean)} rows)")

    # Queue-Times
    qt_clean = clean_queue_times(args.queue)
    qt_out = os.path.join(args.outdir, "queue_times_clean.csv")
    qt_clean.to_csv(qt_out, index=False)
    print(f"Wrote {qt_out} ({len(qt_clean)} rows)")

if __name__ == "__main__":
    main()

