import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

def build_day_features(qt: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate queue_times snapshots to per-day numeric features.
    Features per date: avg_wait, med_wait, p95_wait, pct_open (+ samples for filtering).
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

    # Stability filter to remove tiny days
    min_samples = max(20, int(g["samples"].quantile(0.25)))
    g = g[g["samples"] >= min_samples].reset_index(drop=True)

    return g[["date", "avg_wait", "med_wait", "p95_wait", "pct_open"]]


def make_labels_from_queue(day: pd.DataFrame, mode: str = "median") -> pd.Series:
    """
    Create 'busy'/'not_busy' labels from p95_wait.
      - mode='median': busy if p95_wait >= median(p95_wait) (balanced)
      - mode='q70'   : busy if p95_wait >= 70th percentile (~top 30% busy)
    """
    thr = day["p95_wait"].median() if mode == "median" else day["p95_wait"].quantile(0.70)
    lab = np.where(day["p95_wait"] >= thr, "busy", "not_busy")
    y = pd.Series(lab, index=day.index, name="label")

    print(f"\nLabel rule: busy if p95_wait >= {thr:.2f} (mode='{mode}')")
    print("Label distribution (all days):")
    print(y.value_counts())
    return y


def prep_boosting_data(test_size=0.2, mode="median", random_state=42):
    qt = pd.read_csv("data/clean/queue_times_clean.csv", parse_dates=["timestamp_utc", "date"])
    qt = qt[["date", "ride", "wait_time", "is_open"]].dropna()
    qt = qt[(qt["wait_time"] >= 0) & (qt["wait_time"] <= 300)]

    day = build_day_features(qt)
    y = make_labels_from_queue(day, mode=mode)

    feature_names = ["avg_wait", "med_wait", "pct_open"]
    X_df = day[feature_names].copy()

    # Save small labeled sample
    sample = X_df.copy()
    sample["label"] = y.values
    sample_path = "data/clean/adaboost_labeled_day_sample.csv"
    sample.head(12).to_csv(sample_path, index=False)

    # save full labeled set
    full_boost_df = day[feature_names].copy()
    full_boost_df["label"] = y.values

    full_boost_df.to_csv("data/clean/adaboost_day_full.csv", index=False)


    # Train/test split
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print("\nTrain labels:", pd.Series(y_train).value_counts().to_dict())
    print("Test  labels:", pd.Series(y_test).value_counts().to_dict())

    # Impute missing values (median) using training fit only
    imputer = SimpleImputer(strategy="median")
    X_train_proc = imputer.fit_transform(X_train_df.values)
    X_test_proc  = imputer.transform(X_test_df.values)

    return X_train_proc, X_test_proc, y_train, y_test, feature_names, X_test_df


# ---------- Plot helpers ----------
def save_conf_mat(y_true, y_pred, out_path):
    labels = ["not_busy", "busy"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(cmap="Blues", ax=ax, colorbar=True)
    ax.set_title("AdaBoost Confusion Matrix (Best Settings)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_accuracy_grid(estimator_list, lr_list, acc_grid, depth, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for i, lr in enumerate(lr_list):
        ax.plot(estimator_list, acc_grid[i], marker="o", label=f"lr={lr}")

    ax.set_xlabel("Number of Estimators")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"AdaBoost Accuracy vs Estimators (base depth={depth})")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(model, feature_names, out_path):
    importances = np.array(model.feature_importances_)
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(range(len(feature_names)), importances[order])
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(np.array(feature_names)[order])
    ax.set_ylabel("Importance")
    ax.set_title("AdaBoost Feature Importances (Best Model)")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_adaboost(X_train, X_test, y_train, y_test, feature_names):
    base_depths = [1, 2, 3]
    n_estimators_list = [10, 25, 50, 100, 150, 200]
    lr_list = [0.05, 0.1, 0.5, 1.0]

    best = {"acc": -1, "depth": None, "n": None, "lr": None, "model": None, "pred": None}

    for depth in base_depths:
        stump = DecisionTreeClassifier(max_depth=depth, random_state=42)
        acc_grid = np.zeros((len(lr_list), len(n_estimators_list)))

        for i, lr in enumerate(lr_list):
            for j, n in enumerate(n_estimators_list):
                model = AdaBoostClassifier(
                    estimator=stump,
                    n_estimators=n,
                    learning_rate=lr,
                    random_state=42
                )
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                acc = accuracy_score(y_test, pred)

                acc_grid[i, j] = acc

                if acc > best["acc"]:
                    best.update({
                        "acc": acc, "depth": depth, "n": n, "lr": lr,
                        "model": model, "pred": pred
                    })

        save_accuracy_grid(
            n_estimators_list, lr_list, acc_grid, depth,
            out_path=f"app/static/figures/adaboost_accuracy_depth{depth}.png"
        )

    # Best model summary
    print("\n=== BEST ADABOOST MODEL ===")
    print(f"Best accuracy: {best['acc']:.3f}")
    print(f"Base tree depth: {best['depth']}")
    print(f"n_estimators: {best['n']}")
    print(f"learning_rate: {best['lr']}")

    print("\nClassification report (best model):")
    print(classification_report(y_test, best["pred"], digits=3, zero_division=0))

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, best["pred"], labels=["not_busy", "busy"]))

    save_conf_mat(
        y_test, best["pred"],
        out_path="app/static/figures/adaboost_confusion_matrix.png"
    )
    save_feature_importance(
        best["model"], feature_names,
        out_path="app/static/figures/adaboost_feature_importance.png"
    )

    return best


def main():
    # Data prep
    X_train, X_test, y_train, y_test, feature_names, X_test_df = prep_boosting_data(
        test_size=0.2,
        mode="median"
    )

    print("\nProcessed numeric feature matrix shape:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)


if __name__ == "__main__":
    main()
