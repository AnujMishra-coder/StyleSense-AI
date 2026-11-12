# Latent Dirichlet Allocation topic modeling on Bag-of-Words

import os
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.decomposition import LatentDirichletAllocation

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

def main():
    bow_path = os.path.join(OUTPUT_DIR, "bow.npz")
    features_path = os.path.join(OUTPUT_DIR, "bow_features.csv")
    if not os.path.exists(bow_path) or not os.path.exists(features_path):
        raise FileNotFoundError("BoW files missing. Run src/numerical.py first.")

    X = load_npz(bow_path)
    feature_names = pd.read_csv(features_path)["Feature"].astype(str).tolist()

    n_topics = 8
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=10,
        learning_method="batch",
    )
    lda.fit(X)

    topic_rows = []
    for topic_idx, component in enumerate(lda.components_):
        top_idx = component.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_idx]
        topic_rows.append({
            "Topic": topic_idx + 1,
            "Top Words": ", ".join(top_words)
        })
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

    pd.DataFrame(topic_rows).to_csv(
        os.path.join(OUTPUT_DIR, "lda_topics.csv"), index=False
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(lda, os.path.join(MODEL_DIR, "lda_model.joblib"))
    print("\nLDA topic modeling completed.")

if __name__ == "__main__":
    main()
