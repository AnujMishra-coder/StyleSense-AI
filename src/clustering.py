
# K-Means clustering on TF-IDF review vectors

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import load_npz
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import davies_bouldin_score, silhouette_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    matrix_path = os.path.join(OUTPUT_DIR, "tfidf.npz")
    features_path = os.path.join(OUTPUT_DIR, "tfidf_features.csv")

    if not os.path.exists(matrix_path) or not os.path.exists(features_path):
        raise FileNotFoundError(
            "TF-IDF files missing. Run src/numerical.py first."
        )

    print("\n======= StyleSense AI: Clustering =======")

    X = load_npz(matrix_path)
    feature_names = pd.read_csv(features_path)["Feature"].astype(str).tolist()

    # Reproducible sample
    rng = np.random.RandomState(42)

    sample_size = min(8000, X.shape[0])
    indices = rng.choice(X.shape[0], size=sample_size, replace=False)

    X_sample = X[indices]

    # Reduce dimensionality
    print("Reducing TF-IDF dimensions using TruncatedSVD...")

    svd = TruncatedSVD(n_components=50, random_state=42)
    reduced = svd.fit_transform(X_sample)

    rows = []

    best_k = None
    best_score = float("inf")
    best_labels = None
    best_model = None

    # Try different cluster counts
    for k in range(2, 10):

        print(f"\nTraining K-Means with K={k}...")

        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            batch_size=512,
            n_init=5
        )

        labels = model.fit_predict(X_sample)

        unique_labels = np.unique(labels)

        # Skip invalid clustering results
        if len(unique_labels) < 2:
            print(f"K={k}: Only one cluster found. Skipping.")
            continue

        # Davies-Bouldin score
        db_score = davies_bouldin_score(reduced, labels)

        # Silhouette score on a reproducible subset
        eval_rng = np.random.default_rng(42)

        sample_indices = eval_rng.choice(
            len(labels),
            size=min(2000, len(labels)),
            replace=False
        )

        sample_data = reduced[sample_indices]
        sample_labels = labels[sample_indices]

        if len(np.unique(sample_labels)) < 2:
            sil = np.nan
            print("Silhouette score skipped: evaluation sample has one cluster.")
        else:
            sil = silhouette_score(sample_data, sample_labels)

        print(
            f"K={k}: DB={db_score:.4f}, "
            f"Silhouette={sil:.4f}"
        )

        rows.append({
            "K": k,
            "Davies-Bouldin Score": db_score,
            "Silhouette Score": sil
        })

        # Select best model using Davies-Bouldin score
        if db_score < best_score:
            best_k = k
            best_score = db_score
            best_labels = labels
            best_model = model

    if best_model is None:
        raise RuntimeError("No valid clustering configuration was found.")

    # Save evaluation results
    results = pd.DataFrame(rows)

    results.to_csv(
        os.path.join(OUTPUT_DIR, "clustering_results.csv"),
        index=False
    )

    # Save top words for each cluster
    cluster_terms = []

    for cluster_id, center in enumerate(best_model.cluster_centers_):

        top_idx = np.argsort(center)[-10:][::-1]

        cluster_terms.append({
            "Cluster": cluster_id,
            "Top Words": ", ".join(
                feature_names[i] for i in top_idx
            )
        })

    pd.DataFrame(cluster_terms).to_csv(
        os.path.join(OUTPUT_DIR, "cluster_top_words.csv"),
        index=False
    )

    # Save cluster labels
    np.save(
        os.path.join(OUTPUT_DIR, "cluster_labels.npy"),
        best_labels
    )

    # Plot evaluation results
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        results["K"],
        results["Davies-Bouldin Score"],
        marker="o",
        label="Davies-Bouldin"
    )

    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Davies-Bouldin Score (lower is better)")
    ax.set_title("K-Means Cluster Evaluation")
    ax.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "clustering_evaluation.png"),
        dpi=220
    )

    plt.close(fig)

    print(f"\nBest K by Davies-Bouldin score: {best_k}")
    print("Clustering outputs saved successfully.")


if __name__ == "__main__":
    main()
