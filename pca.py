import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def build_day_features(qt: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to per-day numeric features."""
    g = (qt.groupby("date", as_index=False)
            .agg(
                avg_wait=("wait_time", "mean"),
                med_wait=("wait_time", "median"),
                p95_wait=("wait_time", lambda s: s.quantile(0.95)),
                pct_open=("is_open", "mean"),
                samples=("wait_time", "count"),
            ))
    g["pct_open"] = g["pct_open"] * 100.0
    # require a minimal sample count to stabilize stats
    min_samples = max(20, int(g["samples"].quantile(0.25)))
    g = g[g["samples"] >= min_samples]
    return g[["date", "avg_wait", "med_wait", "p95_wait", "pct_open"]]

def main():
    # Load cleaned queue-times
    qt = pd.read_csv("data/clean/queue_times_clean.csv", parse_dates=["timestamp_utc", "date"])
    qt = qt[["date", "ride", "wait_time", "is_open"]].dropna()
    qt = qt[(qt["wait_time"] >= 0) & (qt["wait_time"] <= 300)]

    day = build_day_features(qt)

    # Join WDW Passport crowd index data
    '''if os.path.exists("data/clean/wdwpassport_clean.csv"):
        wdw = pd.read_csv("data/clean/wdwpassport_clean.csv", parse_dates=["date"])
        day = day.merge(wdw[["date", "crowd_index"]], on="date", how="left")'''

    # Choose features for PCA
    # Start simple; add 'crowd_index' if present:
    candidates = ["avg_wait", "p95_wait", "med_wait", "pct_open"]
    if "crowd_index" in day.columns:
        candidates.append("crowd_index")

    feats = day[candidates].copy()

    # save out sample of data for webpage
    sample_path = os.path.join("data/clean", "pca_input_sample.csv")
    feats.head(10).to_csv(sample_path, index=False)

    # Normalize
    scaler = StandardScaler()
    Xs = scaler.fit_transform(feats.values)

    # Run PCA
    pca = PCA(n_components=None, random_state=42)
    Xp = pca.fit_transform(Xs)

    # Save PCA outputs
    comps = pd.DataFrame(
        Xp, columns=[f"PC{i+1}" for i in range(Xp.shape[1])]
    )
    out = pd.concat([day[["date"]].reset_index(drop=True), comps], axis=1)
    out_path = os.path.join("data/clean", "day_pca_components.csv")
    out.to_csv(out_path, index=False)

    # Loadings (feature contributions to each PC)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=candidates,
        columns=[f"PC{i+1}" for i in range(Xp.shape[1])]
    )
    load_path = os.path.join("data/clean", "day_pca_loadings.csv")
    loadings.to_csv(load_path)

    # Scree plot (variance explained)
    evr = pca.explained_variance_ratio_
    plt.figure(figsize=(7,4))
    plt.plot(range(1, len(evr)+1), evr, marker="o")
    plt.xticks(range(1, len(evr)+1))
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Scree Plot (Day-level features)")
    plt.grid(True, linewidth=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig("/Users/alysaderks/repos/machine_learning_website/app/static/figures/pca_scree.png")
    plt.show()

    # 2D scatter of PC1 vs PC2
    if Xp.shape[1] >= 2:
        plt.figure(figsize=(6,6))
        plt.scatter(Xp[:,0], Xp[:,1], s=18, alpha=0.8)
        plt.xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
        plt.ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
        plt.title("PCA of Disney Day Features (PC1 vs PC2)")
        plt.grid(True, linewidth=0.3, linestyle="--")
        plt.tight_layout()
        plt.savefig("/Users/alysaderks/repos/machine_learning_website/app/static/figures/pca_scatter.png")
        plt.show()

    # choose k components to keep
    cum = np.cumsum(evr)
    k = int(np.searchsorted(cum, 0.90) + 1)
    print(f"Suggested # of PCs for ≥90% variance: k={k}")

if __name__ == "__main__":
    main()