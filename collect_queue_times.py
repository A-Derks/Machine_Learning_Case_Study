import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone

import requests

HDRS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124 Safari/537.36"
    )
}

def fetch_park(park_id: int):
    url = f"https://queue-times.com/parks/{park_id}/queue_times.json"
    r = requests.get(url, headers=HDRS, timeout=15)
    r.raise_for_status()
    return r.json()

def flatten(obj, park_id, ts_iso):
    rows = []
    for land in obj.get("lands", []):
        lname = land.get("name")
        for ride in land.get("rides", []):
            rows.append({
                "timestamp_utc": ts_iso,
                "park_id": park_id,
                "land": lname,
                "ride": ride.get("name"),
                "wait_time": ride.get("wait_time"),
                "is_open": ride.get("is_open"),
                "last_updated": ride.get("last_updated"),
            })
    return rows

def compute_out_path(args, now_utc: datetime) -> str:
    """
    Decide where to write based on --out_mode.
      - single: write to args.out (exact path)
      - daily:  data/<basename>_<YYYY-MM-DD>.csv
      - hourly: data/<basename>_<YYYY-MM-DD_HH>.csv
    """
    if args.out_mode == "single":
        if not args.out:
            raise ValueError("--out is required for out_mode=single")
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        return args.out

    # For rotating modes we use directory + basename
    out_dir = args.out_dir or "data"
    os.makedirs(out_dir, exist_ok=True)

    base = args.basename or f"queue_times_park{args.park_id}"
    if args.out_mode == "daily":
        stamp = now_utc.strftime("%Y-%m-%d")
    elif args.out_mode == "hourly":
        stamp = now_utc.strftime("%Y-%m-%d_%H")
    else:
        raise ValueError("out_mode must be one of: single, daily, hourly")

    return os.path.join(out_dir, f"{base}_{stamp}.csv")

def write_rows(path: str, rows: list):
    header = ["timestamp_utc","park_id","land","ride","wait_time","is_open","last_updated"]
    file_exists = os.path.exists(path)
    # Write atomically to reduce risk of partial writes
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    # Append tmp to real file (or rename if new)
    if not file_exists:
        os.replace(tmp_path, path)
    else:
        # append tmp content to existing file, then remove tmp
        with open(path, "a", newline="", encoding="utf-8") as out_f, \
             open(tmp_path, "r", encoding="utf-8") as in_f:
            for line in in_f:
                out_f.write(line)
        os.remove(tmp_path)

def run_once(args):
    now = datetime.now(timezone.utc)
    ts_iso = now.isoformat()
    try:
        obj = fetch_park(args.park_id)
        rows = flatten(obj, args.park_id, ts_iso)
        out_path = compute_out_path(args, now)
        write_rows(out_path, rows)
        print(f"[{ts_iso}] wrote {len(rows)} rows -> {out_path}")
    except Exception as e:
        print(f"ERROR [{ts_iso}]: {e}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(
        description="Collect Disney queue times with optional file rotation."
    )
    ap.add_argument("--park_id", type=int, required=True,
                    help="e.g., 5=EPCOT, 6=Magic Kingdom, 7=Hollywood Studios, 8=Animal Kingdom")

    # Rotation / output options
    ap.add_argument("--out_mode", default="single", choices=["single", "daily", "hourly"],
                    help="single=one file; daily=one file per day; hourly=one file per hour")
    ap.add_argument("--out", help="Output CSV when out_mode=single (exact filepath)")
    ap.add_argument("--out_dir", help="Directory for rotating modes (default: data)")
    ap.add_argument("--basename", help="Base filename for rotating modes (default: queue_times_park<id>)")

    # Looping / scheduling
    ap.add_argument("--interval", type=int, default=900, help="Seconds between polls when looping (default 900=15min)")
    ap.add_argument("--once", action="store_true", help="Fetch only once and exit")

    args = ap.parse_args()

    if args.once:
        run_once(args)
        return

    # Continuous loop
    while True:
        run_once(args)
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
