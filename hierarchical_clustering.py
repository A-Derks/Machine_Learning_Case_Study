#!/usr/bin/env python3
"""
Hierarchical clustering (cosine) on Disney day-level features.

Features: per-day avg wait & 95th percentile wait
Clustering: Agglomerative (average linkage) with cosine distance
Viz: dendrogram + clustered scatter; automatic k via silhouette (cosine)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# ---------- Config: edit path if needed ----------
CSV = "/Users/alysaderks/repos/machine_learning_website/data/clean/queue_times_clean.csv"


def build_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-day features:
      - avg_wait: mean wait across all rides/snapshots that day
      - p95_wait: 95th percentile across all rides/snapshots that day
    """
    g = (df.groupby("date", as_index=False).agg(avg_wait=("wait_time", "mean"), p95_wait=("wait_time", lambda s: s.quantile(0.95)), samples=("wait_time", "count")))
    return g[["date", "avg_wait", "p95_wait"]].sort_values("date")


# Silhouette method using cosine similarity
def pick_k_silhouette_cosine(X: np.ndarray, k_min=2, k_max=10) -> tuple[int, dict]:
    """
    Try k in [k_min, k_max] with AgglomerativeClustering (average, cosine).
    Returns (best_k, scores_dict).
    """
    scores = {}
    for k in range(k_min, k_max + 1):
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage="average",
            metric="cosine",
        )
        labels = model.fit_predict(X)
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels, metric="cosine")
        else:
            score = -1
        scores[k] = score
        print(f"k={k}: silhouette(cosine) = {score:.3f}")
    best_k = max(scores, key=scores.get)
    print(f"\n Best k by silhouette (cosine): {best_k} (score={scores[best_k]:.3f})")
    return best_k, scores


# ---------- Plots ----------
def plot_silhouette_scores(scores: dict):
    ks = list(scores.keys())
    vals = [scores[k] for k in ks]
    plt.figure(figsize=(7,5))
    plt.plot(ks, vals, marker="o", linestyle="--")
    plt.xticks(ks)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score (cosine)")
    plt.title("Silhouette Analysis (Agglomerative, cosine)")
    plt.grid(True, linewidth=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig("/Users/alysaderks/repos/machine_learning_website/app/static/figures/silhouette_scores.png")
    plt.show()


def plot_dendrogram_cosine(X: np.ndarray, labels=None):
    """
    Build a dendrogram using cosine distance and average linkage.
    """
    # pairwise cosine distance matrix
    D = pdist(X, metric="cosine")
    Z = linkage(D, method="average")
    plt.figure(figsize=(10, 5))
    dendrogram(Z, labels=labels, leaf_rotation=90)
    plt.title("Hierarchical Clustering Dendrogram (average linkage, cosine)")
    plt.xlabel("Day index" if labels is None else "Date")
    plt.ylabel("Cosine distance")
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
        # centers should be in ORIGINAL UNITS: columns [avg_wait, p95_wait]
        plt.scatter(centers[:, 0], centers[:, 1], s=160, marker="X",
                    edgecolor="k", linewidths=1.0, label="Centers")

    plt.xlabel("Daily Average Wait (min)")
    plt.ylabel("Daily 95th Percentile Wait (min)")
    plt.title(f"Hierarchical Clusters (cosine) {title_suffix}")
    plt.grid(True, linewidth=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"/Users/alysaderks/repos/machine_learning_website/app/static/figures/hierarchical_clustering_{title_suffix}.png")
    plt.show()



def main():
    # Load cleaned queue-times
    df = pd.read_csv(CSV, parse_dates=["date"])
    df = df[["date", "ride", "wait_time"]].dropna()
    df = df[(df["wait_time"] >= 0) & (df["wait_time"] <= 300)]

    # Build day-level features
    feats = build_day_features(df)  # columns: date, avg_wait, p95_wait

    X = feats[["avg_wait", "p95_wait"]].to_numpy(dtype=float)
    X_norm = normalize(X, norm="l2")  # L2 normalization preserves cosine geometry

    # 1) Dendrogram (average linkage, cosine)
    Z = plot_dendrogram_cosine(X_norm, labels=feats["date"].dt.strftime("%Y-%m-%d").tolist())

    # 2) Pick k via silhouette (cosine)
    best_k, scores = pick_k_silhouette_cosine(X_norm, k_min=2, k_max=10)
    plot_silhouette_scores(scores)

    # 3) Fit Hierarchical (Agglomerative) with best k (average-link, cosine)
    model = AgglomerativeClustering(n_clusters=best_k, linkage="average", metric="cosine")
    labels = model.fit_predict(X_norm)

    # centers computed in ORIGINAL units (X), not normalized X_norm:
    centers = np.vstack([X[labels == c].mean(axis=0) for c in sorted(np.unique(labels))])

    # Use DataFrame directly
    plot_clusters(feats[["avg_wait", "p95_wait"]], labels, centers=centers, title_suffix=f"(k={best_k})")

    # If you prefer using the features DataFrame directly (cleaner labels):
    plot_clusters(feats[["avg_wait", "p95_wait"]].assign(cluster=labels), labels, centers=centers, title_suffix=f"(k={best_k})")


if __name__ == "__main__":
    main()
