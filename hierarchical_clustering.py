import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram

CSV = "/Users/alysaderks/repos/machine_learning_website/data/clean/queue_times_clean.csv"

def build_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-day features:
      - avg_wait: mean wait across all rides/snapshots that day
      - p95_wait: 95th percentile across all rides/snapshots that day
    """
    g = (
        df.groupby("date", as_index=False)
          .agg(
              avg_wait=("wait_time", "mean"),
              p95_wait=("wait_time", lambda s: s.quantile(0.95)),
              samples=("wait_time", "count"),
          )
    )
    return g[["date", "avg_wait", "p95_wait"]].sort_values("date")


# Silhouette method using Euclidean distance
def pick_k_silhouette_euclidean(X: np.ndarray, k_min=2, k_max=10) -> tuple[int, dict]:
    """
    Try k in [k_min, k_max] with AgglomerativeClustering (average linkage, euclidean).
    Returns (best_k, scores_dict).
    """
    scores = {}
    for k in range(k_min, k_max + 1):
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage="average",
            metric="euclidean",
        )
        labels = model.fit_predict(X)
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels, metric="euclidean")
        else:
            score = -1
        scores[k] = score
        print(f"k={k}: silhouette(euclidean) = {score:.3f}")

    best_k = max(scores, key=scores.get)
    print(f"\nBest k by silhouette (euclidean): {best_k} (score={scores[best_k]:.3f})")
    return best_k, scores


# ---------- Plots ----------
def plot_silhouette_scores(scores: dict):
    ks = list(scores.keys())
    vals = [scores[k] for k in ks]

    plt.figure(figsize=(7,5))
    plt.plot(ks, vals, marker="o", linestyle="--")
    plt.xticks(ks)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score (euclidean)")
    plt.title("Silhouette Analysis (Agglomerative, euclidean)")
    plt.grid(True, linewidth=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig("/Users/alysaderks/repos/machine_learning_website/app/static/figures/silhouette_scores.png")
    plt.show()


def plot_dendrogram_euclidean(X: np.ndarray, labels=None):
    """
    Build a dendrogram using Euclidean distance and average linkage.
    """
    D = pdist(X, metric="euclidean")      # condensed distance matrix
    Z = linkage(D, method="average")     # average linkage on euclidean distances

    plt.figure(figsize=(10, 5))
    dendrogram(Z, labels=labels, leaf_rotation=90)
    plt.title("Hierarchical Clustering Dendrogram (average linkage, euclidean)")
    plt.xlabel("Day index" if labels is None else "Date")
    plt.ylabel("Euclidean distance")
    plt.tight_layout()
    plt.savefig("/Users/alysaderks/repos/machine_learning_website/app/static/figures/dendrogram.png")
    plt.show()
    return Z


def plot_clusters(features_df: pd.DataFrame, labels: np.ndarray, centers=None, title_suffix=""):
    """
    Scatter of p95 vs avg colored by cluster label.
    Expects a DataFrame with columns: ['avg_wait', 'p95_wait'].
    """
    df = features_df.copy()
    df["cluster"] = labels

    plt.figure(figsize=(7, 6))
    for c, sub in df.groupby("cluster"):
        plt.scatter(sub["avg_wait"], sub["p95_wait"], s=18, alpha=0.8, label=f"Cluster {c}")

    if centers is not None:
        plt.scatter(
            centers[:, 0], centers[:, 1],
            s=160, marker="X", edgecolor="k", linewidths=1.0, label="Centers"
        )

    plt.xlabel("Daily Average Wait (min)")
    plt.ylabel("Daily 95th Percentile Wait (min)")
    plt.title(f"Hierarchical Clusters (euclidean) {title_suffix}")
    plt.grid(True, linewidth=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"/Users/alysaderks/repos/machine_learning_website/app/static/figures/hierarchical_clustering_{title_suffix}.png"
    )
    plt.show()


def main():
    # Load cleaned queue-times
    df = pd.read_csv(CSV, parse_dates=["date"])
    df = df[["date", "ride", "wait_time"]].dropna()
    df = df[(df["wait_time"] >= 0) & (df["wait_time"] <= 300)]

    # Build day-level features
    feats = build_day_features(df)

    # Feature matrix
    X = feats[["avg_wait", "p95_wait"]].to_numpy(dtype=float)

    # Standardize for Euclidean clustering
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # 1) Dendrogram (average linkage, euclidean)
    plot_dendrogram_euclidean(
        X_std,
        labels=feats["date"].dt.strftime("%Y-%m-%d").tolist()
    )

    # 2) Pick k via silhouette (euclidean)
    best_k, scores = pick_k_silhouette_euclidean(X_std, k_min=2, k_max=10)
    plot_silhouette_scores(scores)

    # 3) Fit Hierarchical with best k
    model = AgglomerativeClustering(
        n_clusters=best_k,
        linkage="average",
        metric="euclidean"
    )
    labels = model.fit_predict(X_std)

    # centers in ORIGINAL units (avg_wait, p95_wait)
    centers = np.vstack([
        X[labels == c].mean(axis=0)
        for c in sorted(np.unique(labels))
    ])

    plot_clusters(
        feats[["avg_wait", "p95_wait"]],
        labels,
        centers=centers,
        title_suffix=f"(k={best_k})"
    )


if __name__ == "__main__":
    main()

