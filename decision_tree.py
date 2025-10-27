import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

DATA_QT = "data/clean/queue_times_clean.csv"
OUT_DIR = os.path.join("app", "static", "figures")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- 1) Aggregate ride snapshots -> day-level features ----------
def build_day_features(qt: pd.DataFrame) -> pd.DataFrame:
    """
    Features per date: avg_wait, med_wait, p95_wait, pct_open (and samples for filtering).
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

    # Stability filter (ensures enough datapoints per day)
    min_samples = max(20, int(g["samples"].quantile(0.25)))
    g = g[g["samples"] >= min_samples].reset_index(drop=True)
    return g[["date", "avg_wait", "med_wait", "p95_wait", "pct_open"]]


# ---------- 2) Make labels directly from queue_times-derived stats ----------
def make_labels_from_queue(day: pd.DataFrame, mode: str = "median") -> pd.Series:
    """
    'busy' / 'not_busy' labels from p95_wait.
      - mode='median': busy if p95_wait >= median (≈ balanced)
      - mode='q70'   : busy if p95_wait >= 70th percentile (~ top 30% busy)
    """
    thr = day["p95_wait"].median() if mode == "median" else day["p95_wait"].quantile(0.70)
    lab = np.where(day["p95_wait"] >= thr, "busy", "not_busy")
    y = pd.Series(lab, index=day.index, name="label")

    print(f"\nLabel rule: busy if p95_wait >= {thr:.2f} (mode='{mode}')")
    print("Label distribution (all days):")
    print(y.value_counts())
    return y


# ---------- 3) Plot helpers ----------
def save_confusion_matrix(y_true, y_pred, labels=("not_busy", "busy"),
                          out_path=os.path.join(OUT_DIR, "dt_confusion_matrix.png")):
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels))

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — Decision Tree")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix -> {os.path.abspath(out_path)}")


def save_feature_importances(model: DecisionTreeClassifier, feature_names,
                             out_path=os.path.join(OUT_DIR, "dt_feature_importance.png")):
    importances = np.array(model.feature_importances_)
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(range(len(feature_names)), importances[order])
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(np.array(feature_names)[order], rotation=0)
    ax.set_ylabel("Importance")
    ax.set_title("Decision Tree — Feature Importances")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved feature importances -> {os.path.abspath(out_path)}")


def save_tree_plot(model: DecisionTreeClassifier, feature_names,
                   out_path=os.path.join(OUT_DIR, "dt_tree.png")):
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_tree(model,
              feature_names=feature_names,
              class_names=["not_busy", "busy"],
              filled=True, rounded=True, proportion=False, impurity=True)
    ax.set_title("Decision Tree Diagram")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved tree diagram -> {os.path.abspath(out_path)}")


# ---------- 4) Main ----------
def main():
    # Load & clean raw snapshots
    qt = pd.read_csv(DATA_QT, parse_dates=["timestamp_utc", "date"])
    qt = qt[["date", "ride", "wait_time", "is_open"]].dropna()
    qt = qt[(qt["wait_time"] >= 0) & (qt["wait_time"] <= 300)]

    # Build day features
    day = build_day_features(qt)

    # Labels from queue_times only (balanced by default)
    y = make_labels_from_queue(day, mode="median")   # or mode="q70"
    features = ["avg_wait", "med_wait", "p95_wait", "pct_open"]
    X_df = day[features].copy()

    # Train/Test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.20, random_state=42, stratify=y
    )
    print("\nTrain labels:", pd.Series(y_train).value_counts().to_dict())
    print("Test  labels:", pd.Series(y_test).value_counts().to_dict())

    # ---- Train Decision Tree ----
    # criterion: "gini" (Gini impurity) or "entropy" (information gain)
    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=3,           # sensible default; adjust as needed
        min_samples_leaf=3,    # helps prevent overfitting on small data
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Optional: quick 5-fold CV score (on training data only)
    cv = StratifiedKFold(n_splits=min(5, len(np.unique(y_train)) + 3), shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_macro")
    print(f"\n5-fold CV F1_macro (train only): mean={cv_scores.mean():.3f}, scores={np.round(cv_scores,3)}")

    # ---- Evaluate on Test ----
    y_pred = clf.predict(X_test)
    labels = ["not_busy", "busy"]
    print("\n=== Decision Tree (test set) ===")
    print(classification_report(y_test, y_pred, labels=labels, digits=3, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred, labels=labels))

    # ---- Save visuals ----
    save_confusion_matrix(y_test, y_pred, labels=labels)
    save_feature_importances(clf, features)
    save_tree_plot(clf, features)

    # Save predictions table (optional)
    out_csv = os.path.join("data", "clean", "dt_test_predictions.csv")
    out_df = X_test.copy()
    out_df["label_true"] = y_test.values
    out_df["label_pred"] = y_pred
    out_df.to_csv(out_csv, index=False)
    print(f"Saved test predictions -> {os.path.abspath(out_csv)}")


if __name__ == "__main__":
    main()
