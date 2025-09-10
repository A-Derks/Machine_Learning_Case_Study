import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CLEAN_DIR = "data/clean"
WDW_PATH = os.path.join(CLEAN_DIR, "wdwpassport_clean.csv")
QT_PATH = os.path.join(CLEAN_DIR, "queue_times_clean.csv")
OUT_DIR = "app/static/figures"

os.makedirs(OUT_DIR, exist_ok=True)

# Load data
wdw = pd.read_csv(WDW_PATH, parse_dates=["date"])
wdw = wdw.dropna(subset=["date"])
wdw["crowd_index"] = pd.to_numeric(wdw["crowd_index"], errors="coerce")

qt = pd.read_csv(QT_PATH, parse_dates=["timestamp_utc", "date"])
qt["wait_time"] = pd.to_numeric(qt["wait_time"], errors="coerce")
qt["is_open"] = qt["is_open"].astype(str).str.strip().str.lower().map(
    {"true": True, "false": False, "1": True, "0": False, "nan": np.nan}
).fillna(False)
qt["hour"] = qt["timestamp_utc"].dt.tz_localize(None).dt.hour

qt_day = qt.groupby("date", as_index=False).agg(
    avg_wait=("wait_time", "mean"),
    med_wait=("wait_time", "median"),
    p95_wait=("wait_time", lambda s: s.quantile(0.95)),
    pct_open=("is_open", "mean"),
    samples=("wait_time", "count")
)
qt_day["pct_open"] = (qt_day["pct_open"] * 100)

qt_ride = qt.groupby("ride", as_index=False).agg(
    avg_wait=("wait_time", "mean"),
    med_wait=("wait_time", "median"),
    p95_wait=("wait_time", lambda s: s.quantile(0.95)),
    samples=("wait_time", "count")
)

qt_hour_ride = qt.groupby(["ride", "hour"], as_index=False).agg(
    avg_wait=("wait_time", "mean")
)

qt_day_ride_max = qt.groupby(["date", "ride"], as_index=False).agg(
    max_wait=("wait_time", "max")
)

# Merge for correlation plots
daily_merge = pd.merge(
    wdw[["date", "crowd_index"]],
    qt_day[["date", "avg_wait", "p95_wait"]],
    on="date",
    how="inner"
).dropna()

def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=140)
    print(f"Saved: {path}")
    plt.show()

# ----------------------------------------------------
# 1) Daily Crowd Index Over Time (Line)
# ----------------------------------------------------
def viz1_crowd_over_time():
    fig = plt.figure()
    ax = fig.gca()
    d = wdw.sort_values("date")
    ax.plot(d["date"], d["crowd_index"])
    ax.set_title("Daily Crowd Index Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Crowd Index (1–10)")
    ax.grid(True, linewidth=0.3)
    save_fig(fig, "01_crowd_index_over_time.png")

# ----------------------------------------------------
# 2) Day of Week Crowd Patterns
# ----------------------------------------------------
def viz2_dow_crowd_bars():
    if wdw.empty:
        print("No WDW Passport data.")
        return

    d = wdw.copy()
    d["dow"] = d["date"].dt.weekday  # 0=Mon,6=Sun
    lab = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    g = (d.groupby("dow", as_index=False)
          .agg(mean_idx=("crowd_index","mean"),
               std_idx =("crowd_index","std"),
               n=("crowd_index","count")))

    fig = plt.figure(figsize=(9, 5))
    ax = fig.gca()
    x = np.arange(7)

    ax.bar(x, g["mean_idx"], yerr=g["std_idx"], capsize=3)
    ax.set_title("Average Crowd Index by Day of Week (±1 SD)")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Crowd Index (1–10)")
    ax.set_xticks(x)
    ax.set_xticklabels(lab)
    ax.set_ylim(0, 10)
    ax.grid(True, axis="y", linewidth=0.3, linestyle="--")

    save_fig(fig, "02_dow_crowd_index_bars.png")

# ----------------------------------------------------
# 3) Average Wait by Ride (Bar; top 20)
# ----------------------------------------------------
def viz3_avg_wait_by_ride(top_n=20):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca()
    rides = qt_ride.sort_values("avg_wait", ascending=False).head(top_n)
    ax.bar(rides["ride"], rides["avg_wait"])
    ax.set_title(f"Top {top_n} Rides by Average Wait")
    ax.set_xlabel("Ride")
    ax.set_ylabel("Average Wait (min)")
    ax.set_xticklabels(rides["ride"], rotation=75, ha="right")
    ax.grid(True, linewidth=0.3, axis="y")
    save_fig(fig, "03_avg_wait_by_ride.png")

# ----------------------------------------------------
# 4) Wait Time Distribution for a ride
# ----------------------------------------------------
def viz4_wait_histogram(ride_name=None):
    fig = plt.figure()
    ax = fig.gca()
    data = qt if ride_name is None else qt[qt["ride"] == ride_name]
    vals = data["wait_time"].dropna().values
    bins = min(50, max(10, int(math.sqrt(len(vals))))) if len(vals) else 10
    ax.hist(vals, bins=bins)
    title = f"Wait Time Distribution ({ride_name})" if ride_name else "Wait Time Distribution (All Rides)"
    ax.set_title(title)
    ax.set_xlabel("Wait Time (min)")
    ax.set_ylabel("Frequency")
    ax.grid(True, linewidth=0.3)
    save_fig(fig, "04_wait_time_histogram.png")

# ----------------------------------------------------
# 5) Hourly Wait Pattern Heatmap
# ----------------------------------------------------
def viz5_hourly_heatmap(top_n=20):
    counts = qt.groupby("ride")["wait_time"].count().sort_values(ascending=False)
    top = list(counts.head(top_n).index)
    sub = qt_hour_ride[qt_hour_ride["ride"].isin(top)]
    pivot = sub.pivot(index="ride", columns="hour", values="avg_wait")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca()
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_title("Hourly Wait Pattern (Top Rides)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Ride")
    ax.set_xticks(range(0, 24))
    ax.set_xticklabels(range(0, 24))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(im, ax=ax, label="Avg Wait (min)")
    save_fig(fig, "05_hourly_wait_heatmap.png")

# ----------------------------------------------------
# 6) Peak vs Average Daily Wait
# ----------------------------------------------------
def viz6_peak_vs_avg():
    # Daily average wait
    avg = qt_day[["date", "avg_wait"]].dropna()

    # Daily peak wait across all rides
    peak_by_day = (qt_day_ride_max
                   .groupby("date", as_index=False)["max_wait"]
                   .max()
                   .rename(columns={"max_wait": "peak_wait"}))

    d = (pd.merge(avg, peak_by_day, on="date", how="inner")
         .dropna()
         .sort_values("date"))

    if d.empty:
        print("No daily avg/peak data to plot.")
        return

    # Auto-aggregate for readability
    n = len(d)
    if n > 365:
        freq = "MS"  # month start
        label_fmt = "%b %Y"
        out_name = "06_peak_vs_avg_monthly_bars.png"
    elif n > 180:
        freq = "W-MON"  # weekly
        label_fmt = "%Y-%m-%d"
        out_name = "06_peak_vs_avg_weekly_bars.png"
    else:
        freq = None
        label_fmt = "%b %d"
        out_name = "06_peak_vs_avg_daily_bars.png"

    if freq:
        d = (d.set_index("date")
             .resample(freq)
             .mean(numeric_only=True)
             .dropna()
             .reset_index())

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(d))
    width = 0.45

    ax.bar(x - width / 2, d["avg_wait"], width=width, label="Daily Average Wait (min)")
    ax.bar(x + width / 2, d["peak_wait"], width=width, label="Daily Peak Wait (min)")

    ax.set_title("Average vs Peak Daily Wait")
    ax.set_xlabel("Date")
    ax.set_ylabel("Wait Time (minutes)")
    ax.set_xticks(x)
    ax.set_xticklabels(d["date"].dt.strftime(label_fmt), rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.3)
    ax.legend(loc="upper left")

    save_fig(fig, out_name)

# ----------------------------------------------------
# 7) Monthly Boxplots
# ----------------------------------------------------
def viz7_monthly_boxplots(use_wait=True):
    fig = plt.figure()
    ax = fig.gca()
    if use_wait:
        tmp = qt_day.copy()
        tmp["month"] = tmp["date"].dt.month
        data = [tmp.loc[tmp["month"] == m, "avg_wait"].dropna().values for m in range(1, 13)]
        ax.boxplot(data, positions=list(range(1, 13)))
        ax.set_title("Monthly Distribution of Daily Average Wait")
        ax.set_ylabel("Daily Avg Wait (min)")
        ax.set_xlabel("Month")
        ax.grid(True, linewidth=0.3, axis="y")
        save_fig(fig, "07_monthly_boxplot_daily_avg_wait.png")
    else:
        tmp = wdw.copy()
        tmp["month"] = tmp["date"].dt.month
        data = [tmp.loc[tmp["month"] == m, "crowd_index"].dropna().values for m in range(1, 13)]
        ax.boxplot(data, positions=list(range(1, 13)))
        ax.set_title("Monthly Distribution of Crowd Index")
        ax.set_ylabel("Crowd Index (1–10)")
        ax.set_xlabel("Month")
        ax.grid(True, linewidth=0.3, axis="y")
        save_fig(fig, "07_monthly_boxplot_crowd_index.png")

# ----------------------------------------------------
# 8) Ride Uptime %
# ----------------------------------------------------
def viz8_uptime_by_ride(top_n=25):
    op = qt.groupby("ride", as_index=False).agg(
        pct_open=("is_open", lambda s: float(s.mean()) * 100.0),
        samples=("is_open", "count")
    )
    op = op[op["samples"] >= max(50, int(op["samples"].quantile(0.5)))]
    op = op.sort_values("pct_open", ascending=False).head(top_n)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca()
    ax.bar(op["ride"], op["pct_open"])
    ax.set_title("Ride Operational Uptime (% of Samples Open)")
    ax.set_xlabel("Ride")
    ax.set_ylabel("Open %")
    ax.set_xticklabels(op["ride"], rotation=75, ha="right")
    ax.grid(True, linewidth=0.3, axis="y")
    save_fig(fig, "08_ride_uptime.png")

# ----------------------------------------------------
# 9) Crowd Index vs Avg Wait
# ----------------------------------------------------
def viz9_crowd_vs_wait():
    if daily_merge.empty:
        print("No overlap between WDW Passport and Queue-Times daily aggregates.")
        return

    d = daily_merge.sort_values("date").copy()

    # Auto-aggregate if too many days
    n = len(d)
    if n > 365:
        freq, label_fmt, out_name = "MS", "%b %Y", "crowd_vs_wait_bar_monthly.png"
    elif n > 180:
        freq, label_fmt, out_name = "W-MON", "%Y-%m-%d", "crowd_vs_wait_bar_weekly.png"
    else:
        freq, label_fmt, out_name = None, "%b %d", "crowd_vs_wait_bar_daily.png"

    if freq:
        d = (
            d.set_index("date")
            .resample(freq)
            .mean(numeric_only=True)
            .dropna(subset=["crowd_index", "avg_wait"])
            .reset_index()
        )

    # Normalize both to [0,1] for fair comparison
    d["crowd_norm"] = d["crowd_index"] / d["crowd_index"].max()
    d["wait_norm"] = d["avg_wait"] / d["avg_wait"].max()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(d))
    width = 0.45

    ax.bar(x - width / 2, d["crowd_norm"], width=width, color="steelblue", label="Crowd Index (scaled)")
    ax.bar(x + width / 2, d["wait_norm"], width=width, color="indianred", label="Avg Wait (scaled)")

    ax.set_title("Crowd Index vs Avg Wait (Scaled)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Scaled Value (0–1)")
    ax.set_xticks(x)
    ax.set_xticklabels(d["date"].dt.strftime(label_fmt), rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.3)
    ax.legend(loc="upper left")

    save_fig(fig, out_name)


if __name__ == "__main__":
    viz1_crowd_over_time()
    viz2_dow_crowd_bars()
    viz3_avg_wait_by_ride(top_n=20)
    viz4_wait_histogram(ride_name=None)
    viz5_hourly_heatmap(top_n=20)
    viz6_peak_vs_avg()
    viz7_monthly_boxplots(use_wait=True)
    viz7_monthly_boxplots(use_wait=False) # uses crowd index
    viz8_uptime_by_ride(top_n=25)
    viz9_crowd_vs_wait()
