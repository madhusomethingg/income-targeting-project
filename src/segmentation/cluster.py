"""
cluster.py
----------
Population segmentation using K-Means clustering.

Two segmentations are produced:
  1. Global  — full population (all income levels)
  2. Premium — restricted to predicted >50K individuals (profit-sim threshold)

Pipeline per segmentation:
  1. Convert encoded feature matrix to scipy sparse CSR for memory efficiency
  2. TruncatedSVD for dimensionality reduction (no scaling — OHE columns are
     binary; scaling distorts meaning and can overweight rare categories)
  3. Elbow + silhouette analysis to visually confirm k choice
     (silhouette computed on unweighted random sample of 5,000 rows for speed)
  4. K-Means clustering with chosen k
  5. Profile each segment using PRE-ENCODED cleaned features (not OHE matrix)
     so that numeric columns like age, wage, capital gains are readable

Design decisions:
  - Two feature matrices used throughout:
      X_cluster : post-OHE encoded (for SVD + KMeans)
      X_profile : pre-OHE cleaned  (for segment profiling — readable features)
  - X_cluster is built by encoding the full X_clean directly using train
    vocabulary from build_features — avoids index misalignment that would
    occur if concatenating shuffled train/val/test splits back together.
  - scipy.sparse.csr_matrix used before SVD — correct for large OHE matrices.
  - Silhouette unweighted and sampled — sklearn silhouette does not
    consistently support sample_weight across versions.
  - Premium subset size reported as weighted population share, not sample %.
  - Full dataset used for clustering — segmentation is unsupervised and
    produces population-level segments, not a generalisation metric.
    This is explicitly not data leakage.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import yaml
import joblib

from pathlib import Path
from typing import Tuple, Dict
from scipy import sparse

from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def reduce_dimensions(
    X_cluster: pd.DataFrame,
    config: dict,
    label: str = "global",
) -> Tuple[np.ndarray, TruncatedSVD]:
    """
    Apply TruncatedSVD for dimensionality reduction.

    No scaling applied — post-OHE features are mostly binary indicators.
    Scaling changes their meaning and can cause rare categories to dominate.

    Input converted to scipy sparse CSR — correct and memory-efficient
    for large OHE feature matrices.

    n_components chosen to explain pca_variance_threshold of variance.
    """
    seg_cfg     = config["segmentation"]
    variance_th = seg_cfg["pca_variance_threshold"]
    seed        = config["random_seed"]

    X_sparse = sparse.csr_matrix(X_cluster.values.astype("float32"))

    max_components = min(100, X_sparse.shape[1] - 1)
    max_components = max(max_components, 2)

    svd_full = TruncatedSVD(n_components=max_components, random_state=seed)
    svd_full.fit(X_sparse)

    cumvar       = np.cumsum(svd_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, variance_th)) + 1
    n_components = max(n_components, 2)

    print(f"[cluster/{label}] SVD: {n_components} components explain "
          f"{cumvar[n_components-1]*100:.1f}% of variance")

    svd       = TruncatedSVD(n_components=n_components, random_state=seed)
    X_reduced = svd.fit_transform(X_sparse)

    return X_reduced, svd


def elbow_and_silhouette(
    X_reduced: np.ndarray,
    config: dict,
    label: str = "global",
) -> None:
    """
    Plot inertia (elbow) and silhouette score across k values.
    Silhouette is unweighted and computed on a random sample of 5,000 rows.
    Saves figure to results/figures/ for visual k confirmation.
    """
    seg_cfg  = config["segmentation"]
    k_range  = range(seg_cfg["n_clusters_range"][0],
                     seg_cfg["n_clusters_range"][1] + 1)
    seed     = config["random_seed"]
    n_init   = seg_cfg.get("n_init", 20)
    max_iter = seg_cfg.get("max_iter", 300)

    inertias    = []
    silhouettes = []

    for k in k_range:
        km     = KMeans(n_clusters=k, n_init=n_init,
                        max_iter=max_iter, random_state=seed)
        labels = km.fit_predict(X_reduced)
        inertias.append(km.inertia_)

        sample_size = min(5000, len(X_reduced))
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_reduced), sample_size, replace=False)
        sil = silhouette_score(X_reduced[idx], labels[idx])
        silhouettes.append(sil)

        print(f"[cluster/{label}] k={k}  inertia={km.inertia_:,.0f}  "
              f"silhouette(sample, unweighted)={sil:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(list(k_range), inertias, "o-", color="steelblue", linewidth=2)
    ax1.set_xlabel("Number of clusters (k)", fontsize=11)
    ax1.set_ylabel("Inertia", fontsize=11)
    ax1.set_title(f"Elbow Curve — {label}", fontsize=12)
    ax1.grid(alpha=0.3)

    ax2.plot(list(k_range), silhouettes, "o-", color="darkorange", linewidth=2)
    ax2.set_xlabel("Number of clusters (k)", fontsize=11)
    ax2.set_ylabel("Silhouette Score (sampled, unweighted)", fontsize=11)
    ax2.set_title(f"Silhouette Scores — {label}", fontsize=12)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = (Path(config["paths"]["results_figures"])
                / f"cluster_selection_{label}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[cluster/{label}] Elbow/silhouette plot saved → {out_path}")


def fit_kmeans(
    X_reduced: np.ndarray,
    n_clusters: int,
    config: dict,
    label: str = "global",
) -> Tuple[KMeans, np.ndarray]:
    """Fit final K-Means with chosen k. Returns model and cluster labels."""
    seg_cfg  = config["segmentation"]
    seed     = config["random_seed"]
    n_init   = seg_cfg.get("n_init", 20)
    max_iter = seg_cfg.get("max_iter", 300)

    km     = KMeans(n_clusters=n_clusters, n_init=n_init,
                    max_iter=max_iter, random_state=seed)
    labels = km.fit_predict(X_reduced)

    print(f"[cluster/{label}] K-Means fitted  k={n_clusters}  "
          f"inertia={km.inertia_:,.0f}")

    unique, counts = np.unique(labels, return_counts=True)
    for cl, ct in zip(unique, counts):
        print(f"[cluster/{label}]   Cluster {cl}: {ct:,} rows "
              f"({ct/len(labels)*100:.1f}%)")

    return km, labels


def build_segment_profiles(
    X_profile: pd.DataFrame,
    labels: np.ndarray,
    weights: np.ndarray,
    y: pd.Series,
    config: dict,
    label: str = "global",
) -> pd.DataFrame:
    """
    Build a profile table for each cluster using PRE-ENCODED cleaned features.

    X_profile must be the cleaned DataFrame BEFORE one-hot encoding so that
    readable numeric columns (age, wage per hour, capital gains etc.) exist.

    Output columns:
      - weighted_size      : population-representative count
      - pct_of_population  : % of total weighted population
      - high_income_rate   : weighted % of >50K per cluster
      - mean_{feature}     : weighted mean of key numeric features
    """
    weights = np.asarray(weights, dtype=float)

    df = X_profile.copy()
    df["_cluster"] = labels
    df["_weight"]  = weights
    df["_label"]   = np.asarray(y)

    # Exact column names from census-bureau.columns (with spaces)
    candidate_cols = [
        "age",
        "wage per hour",
        "capital gains",
        "capital losses",
        "dividends from stocks",
        "weeks worked in year",
        "num persons worked for employer",
    ]
    num_cols = [c for c in candidate_cols if c in df.columns]

    if not num_cols:
        print(f"[cluster/{label}] WARNING: no profiling numeric columns found. "
              f"Ensure X_profile is the pre-encoded feature matrix.")

    profiles     = []
    total_weight = weights.sum()

    for cl in sorted(df["_cluster"].unique()):
        mask    = df["_cluster"] == cl
        subset  = df[mask]
        w_sub   = subset["_weight"].values.astype(float)
        w_total = w_sub.sum()

        row = {
            "cluster"           : cl,
            "weighted_size"     : round(w_total, 0),
            "pct_of_population" : round(w_total / total_weight * 100, 2),
            "high_income_rate"  : round(
                np.average(subset["_label"].values, weights=w_sub) * 100, 2
            ),
        }

        for col in num_cols:
            row[f"mean_{col}"] = round(
                np.average(subset[col].values, weights=w_sub), 2
            )

        profiles.append(row)

    profiles_df = pd.DataFrame(profiles).set_index("cluster")

    out_path = (Path(config["paths"]["results_tables"])
                / f"segment_profiles_{label}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profiles_df.to_csv(out_path)
    print(f"\n[cluster/{label}] Segment profiles saved → {out_path}")
    print(profiles_df.to_string())

    return profiles_df


def run_segmentation(
    X_cluster: pd.DataFrame,
    X_profile: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    config: dict,
    model,
    optimal_threshold: float,
) -> Dict:
    """
    Run both global and premium segmentations.

    Parameters
    ----------
    X_cluster         : post-OHE encoded features (for SVD + KMeans)
    X_profile         : pre-OHE cleaned features  (for segment profiling)
    y                 : labels (full dataset, aligned with X_cluster)
    w                 : weights (full dataset, aligned with X_cluster)
    config            : project config
    model             : fitted XGBoost (for premium filter)
    optimal_threshold : from profit simulation
    """
    seg_cfg = config["segmentation"]
    results = {}

    # ==================================================================
    # 1. GLOBAL SEGMENTATION — full population
    # ==================================================================
    print("\n" + "="*60)
    print("GLOBAL SEGMENTATION")
    print("="*60)

    X_reduced_g, svd_g = reduce_dimensions(X_cluster, config, label="global")
    elbow_and_silhouette(X_reduced_g, config, label="global")

    n_global       = seg_cfg["n_clusters_global"]
    km_g, labels_g = fit_kmeans(X_reduced_g, n_global, config, label="global")

    profiles_g = build_segment_profiles(
        X_profile, labels_g, np.asarray(w, dtype=float),
        y, config, label="global"
    )

    models_dir = Path(config["paths"]["results_models"])
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(km_g, models_dir / "kmeans_global.joblib")
    print(f"[cluster/global] Model saved → {models_dir / 'kmeans_global.joblib'}")

    results["global"] = {
        "labels"  : labels_g,
        "profiles": profiles_g,
        "model"   : km_g,
    }

    # ==================================================================
    # 2. PREMIUM SEGMENTATION — predicted >50K only
    # ==================================================================
    print("\n" + "="*60)
    print("PREMIUM SEGMENTATION (predicted >50K)")
    print("="*60)

    y_proba      = model.predict_proba(X_cluster)[:, 1]
    premium_mask = y_proba >= optimal_threshold

    X_cluster_prem = X_cluster[premium_mask].reset_index(drop=True)
    X_profile_prem = X_profile[premium_mask].reset_index(drop=True)
    y_prem         = y[premium_mask].reset_index(drop=True)
    w_prem         = w[premium_mask].reset_index(drop=True)

    # Population share — weighted, not raw sample row %
    pop_share = w[premium_mask].sum() / w.sum()
    print(f"[cluster/premium] Premium subset: {premium_mask.sum():,} rows "
          f"({pop_share*100:.1f}% of weighted population)")

    X_reduced_p, svd_p = reduce_dimensions(X_cluster_prem, config, label="premium")
    elbow_and_silhouette(X_reduced_p, config, label="premium")

    n_premium      = seg_cfg["n_clusters_premium"]
    km_p, labels_p = fit_kmeans(X_reduced_p, n_premium, config, label="premium")

    profiles_p = build_segment_profiles(
        X_profile_prem, labels_p, np.asarray(w_prem, dtype=float),
        y_prem, config, label="premium"
    )

    joblib.dump(km_p, models_dir / "kmeans_premium.joblib")
    print(f"[cluster/premium] Model saved → {models_dir / 'kmeans_premium.joblib'}")

    results["premium"] = {
        "labels"  : labels_p,
        "profiles": profiles_p,
        "model"   : km_p,
        "mask"    : premium_mask,
    }

    return results


if __name__ == "__main__":
    # Run from project root:
    # python -m src.segmentation.cluster
    from src.data.loader import load_raw_data
    from src.data.cleaner import clean
    from src.features.engineer import build_features

    config = load_config("config.yaml")
    raw_df = load_raw_data(config)

    # X_clean: pre-encoded cleaned features — used for profiling
    X_clean, y, w = clean(raw_df, config)

    # Run build_features to get train vocab (median_map, feature_columns)
    data = build_features(X_clean.copy(), y.copy(), w.copy(), config)

    # Build X_cluster: encode the FULL X_clean using train vocabulary
    # This avoids index misalignment from concatenating shuffled train/val/test
    X_seg = X_clean.copy().reset_index(drop=True)
    y_seg = y.reset_index(drop=True)
    w_seg = w.reset_index(drop=True)

    # Apply train medians to full dataset
    for col, med in data["median_map"].items():
        if col in X_seg.columns:
            X_seg[col] = X_seg[col].fillna(med)

    # Apply log1p transform
    skewed_cols = config["data"].get("skewed_numeric_columns", [])
    for col in [c for c in skewed_cols if c in X_seg.columns]:
        X_seg[col] = np.log1p(X_seg[col].clip(lower=0))

    # X_profile = pre-encoding state (for readable segment profiles)
    X_profile_all = X_seg.copy()

    # One-hot encode using same settings as engineer.py
    cat_cols   = X_seg.select_dtypes(include="object").columns.tolist()
    drop_first = config["features"].get("drop_first", False)
    X_cluster_all = pd.get_dummies(X_seg, columns=cat_cols, drop_first=drop_first)

    # Sanitize column names (same as engineer.py sanitize_columns)
    X_cluster_all.columns = (
        X_cluster_all.columns.str.replace("[", "_", regex=False)
                              .str.replace("]", "_", regex=False)
                              .str.replace("<", "_", regex=False)
                              .str.replace(">", "_", regex=False)
                              .str.replace(" ", "_", regex=False)
    )
    # Reindex to match train feature space exactly
    X_cluster_all = X_cluster_all.reindex(
        columns=data["feature_columns"], fill_value=0
    )

    print(f"[cluster] X_cluster shape : {X_cluster_all.shape}")
    print(f"[cluster] X_profile shape : {X_profile_all.shape}")

    # Load champion model and optimal threshold
    models_dir = Path(config["paths"]["results_models"])
    xgb_model  = joblib.load(models_dir / "xgboost.joblib")

    threshold_df      = pd.read_csv("results/tables/optimal_threshold.csv")
    optimal_threshold = float(threshold_df["optimal_threshold"].iloc[0])
    print(f"[cluster] Using optimal threshold = {optimal_threshold:.4f}")

    results = run_segmentation(
        X_cluster_all, X_profile_all,
        y_seg, w_seg,
        config, xgb_model, optimal_threshold
    )