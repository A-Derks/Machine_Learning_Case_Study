import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def build_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-day features:
      - avg_wait: mean wait across all rides/snapshots that day
      - p95_wait: 95th percentile across all rides/snapshots that day
    """
    g = (df.groupby("date", as_index=False).agg(avg_wait=("wait_time", "mean"), p95_wait=("wait_time", lambda s: s.quantile(0.95)), samples=("wait_time", "count")))
    return g[["date", "avg_wait", "p95_wait"]].sort_values("date")

def silhouette_analysis(X, k_min=2, k_max=10):
    """Compute silhouette scores for k in [k_min, k_max]."""
    scores = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X)
        if len(set(labels)) > 1:  # silhouette requires >1 cluster
            score = silhouette_score(X, labels)
            scores[k] = score
            print(f"k={k}, silhouette={score:.3f}")
        else:
            scores[k] = -1

    # Plot silhouette scores
    plt.figure(figsize=(7, 5))
    ks = list(scores.keys())
    vals = list(scores.values())
    plt.plot(ks, vals, marker="o", linestyle="--")
    plt.xticks(ks)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette Analysis for Optimal k")
    plt.grid(True, linewidth=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig("/Users/alysaderks/repos/machine_learning_website/app/static/figures/silhouette_method.png")
    plt.show()

    best_k = max(scores, key=scores.get)
    print(f"\n Best k by silhouette: {best_k} (score={scores[best_k]:.3f})")
    return best_k

def k_means_clustering(features, n_clusters, max_iter=100):
    """Run local implementation of kmeans."""
    # Standardize the data
    X = features.values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=20)
    labels = km.fit_predict(Xs)

    # Convert centers back to original units (minutes)
    centers_orig = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=features.columns
    )

    return km, labels, centers_orig


def main():
    # Import the data
    df = pd.read_csv(
        "/Users/alysaderks/repos/machine_learning_website/data/clean/queue_times_clean.csv",
        parse_dates=["timestamp_utc", "date"]
    )

    # Keep needed cols, ensure numeric & sane bounds
    df = df[["date", "ride", "wait_time"]].copy()


    # Build daily features: avg_wait & p95_wait
    feature_cols = ["avg_wait", "p95_wait"]
    day_feats = build_day_features(df)  # columns: date, avg_wait, p95_wait

    scaler = StandardScaler()
    Xs = scaler.fit_transform(day_feats[feature_cols].values)
    best_k = silhouette_analysis(Xs, k_min=2, k_max=10)

    plt.figure(figsize=(7, 6))
    plt.scatter(day_feats["avg_wait"], day_feats["p95_wait"], s=18, alpha=0.7)
    plt.xlabel("Daily Average Wait (min)")
    plt.ylabel("Daily 95th Percentile Wait (min)")
    plt.title("Daily Avg vs 95th Percentile Wait (Unclustered)")
    plt.grid(True, linewidth=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig("/Users/alysaderks/repos/machine_learning_website/app/static/figures/k_means_unclustered.png")
    plt.show()

    # Run local implementation of kmeans with best k
    km, labels, centers_orig = k_means_clustering(day_feats[feature_cols], n_clusters=best_k)

    # Attach labels for inspection/saving
    day_feats["cluster"] = labels

    # plot clustered data
    plt.figure(figsize=(7, 6))
    # color by cluster
    for c in sorted(np.unique(labels)):
        sub = day_feats[day_feats["cluster"] == c]
        plt.scatter(sub["avg_wait"], sub["p95_wait"], s=18, alpha=0.8, label=f"Cluster {c}")
    # plot centers
    plt.scatter(centers_orig["avg_wait"], centers_orig["p95_wait"], s=160, marker="X", edgecolor="k", linewidths=1.0,
                label="Centers")
    plt.xlabel("Daily Average Wait (min)")
    plt.ylabel("Daily 95th Percentile Wait (min)")
    plt.title("KMeans Clusters of Days (Avg vs P95) k=2")
    plt.grid(True, linewidth=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/Users/alysaderks/repos/machine_learning_website/app/static/figures/k_means_clustered_k_2.png")
    plt.show()


if __name__ == "__main__":
    main()