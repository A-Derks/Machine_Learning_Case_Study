import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import KBinsDiscretizer

def build_day_features(qt: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate queue_times snapshots to per-day numeric features.
    Features: avg_wait, med_wait, p95_wait, pct_open, samples (for filtering).
    """
    g = (qt.groupby("date", as_index=False)
           .agg(
               avg_wait=("wait_time", "mean"),
               med_wait=("wait_time", "median"),
               p95_wait=("wait_time", lambda s: s.quantile(0.95)),
               pct_open=("is_open", "mean"),
               samples=("wait_time", "count"),
           ))
    g["pct_open"] = g["pct_open"] * 100.0

    # Stability filter
    min_samples = max(20, int(g["samples"].quantile(0.25)))
    g = g[g["samples"] >= min_samples].reset_index(drop=True)

    return g[["date", "avg_wait", "med_wait", "p95_wait", "pct_open"]]


def make_labels_from_queue(day: pd.DataFrame, mode: str = "median") -> pd.Series:
    """
    Create 'busy'/'not_busy' labels
      - mode='median': busy if p95_wait >= median(p95_wait)  (≈ balanced 50/50)
      - mode='q70':    busy if p95_wait >= 70th percentile   (~ top 30% busy)
    """
    if mode == "q70":
        thr = day["p95_wait"].quantile(0.70)
    else:
        thr = day["p95_wait"].median()

    lab = np.where(day["p95_wait"] >= thr, "busy", "not_busy")
    y = pd.Series(lab, index=day.index, name="label")

    print(f"\nLabel rule: busy if p95_wait >= {thr:.2f} (mode='{mode}')")
    print("Label distribution (all days):")
    print(y.value_counts())
    return y

def save_train_test_split_image(y_train, y_test, out_path="app/static/figures/train_test_split.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Class order (ensure both appear even if one count is 0)
    classes = ["not_busy", "busy"]
    train_counts = [int((pd.Series(y_train)==c).sum()) for c in classes]
    test_counts  = [int((pd.Series(y_test)==c).sum())  for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(x - width/2, train_counts, width, label="Train")
    ax.bar(x + width/2, test_counts,  width, label="Test")

    ax.set_xticks(x, classes)
    ax.set_ylabel("Number of days")
    ax.set_title("Train/Test Split by Class")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    # Write exact counts above bars
    for i, v in enumerate(train_counts):
        ax.text(x[i] - width/2, v + 0.15, str(v), ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(test_counts):
        ax.text(x[i] + width/2, v + 0.15, str(v), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"Saved train/test split image -> {out_path}")

def save_confusion_matrix_plot(y_true, y_pred, labels=None, out_path="app/static/figures/confusion_matrix_nb.png"):
    """Save a styled confusion matrix plot as a PNG."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    # Plot
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    plt.title("Confusion Matrix — Naive Bayes Wait Time Model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    abs_path = os.path.abspath(out_path)
    print(f"Saved confusion matrix image -> {abs_path}")


def main():
    # Load & clean data
    qt = pd.read_csv("data/clean/queue_times_clean.csv", parse_dates=["timestamp_utc", "date"])
    qt = qt[["date", "ride", "wait_time", "is_open"]].dropna()
    qt = qt[(qt["wait_time"] >= 0) & (qt["wait_time"] <= 300)]

    # Build per-day features
    day = build_day_features(qt)

    # Make supervised labels
    y = make_labels_from_queue(day, mode="median")

    # Save a small labeled sample for report
    sample = day.copy()
    sample["label"] = y
    sample_path = os.path.join("data/clean", "nb_labeled_day_sample.csv")
    sample.head(12).to_csv(sample_path, index=False)
    print(f"Saved labeled sample -> {sample_path}")

    # Feature selection
    features = ["avg_wait", "med_wait", "p95_wait", "pct_open"]
    X_df = day[features].copy()

    # Train/Test split
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.20, random_state=42, stratify=y
    )
    print("\nTrain labels:", pd.Series(y_train).value_counts().to_dict())
    print("Test  labels:", pd.Series(y_test).value_counts().to_dict())

    save_train_test_split_image(y_train, y_test)

    # Naive Bayes
    n_bins = [min(8, max(2, X_train_df[c].nunique() - 1)) for c in X_train_df.columns]

    kb_kwargs = dict(n_bins=n_bins, encode="onehot", strategy="quantile")
    kb = KBinsDiscretizer(**kb_kwargs, quantile_method="averaged_inverted_cdf")

    Xtr_m = kb.fit_transform(X_train_df.values)
    Xte_m = kb.transform(X_test_df.values)

    clf = MultinomialNB()
    clf.fit(Xtr_m, y_train)
    y_pred = clf.predict(Xte_m)

    # Evaluation
    labels = ["not_busy", "busy"]
    print("\n=== Multinomial Naive Bayes (discretized features) ===")
    print(classification_report(y_test, y_pred, labels=labels, digits=3, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred, labels=labels))

    save_confusion_matrix_plot(y_test, y_pred, labels=labels)

    # Save predictions
    test_out = X_test_df.copy()
    test_out["label_true"] = y_test.values
    test_out["label_pred"] = y_pred
    out_csv = os.path.join("data/clean", "nb_test_predictions.csv")
    test_out.to_csv(out_csv, index=False)
    print(f"\nSaved predictions -> {out_csv}")


if __name__ == "__main__":
    main()

