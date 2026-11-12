
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import load_npz
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import classification_report


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"

DATA_PATH = OUTPUT_DIR / "processed_reviews.csv"
TFIDF_PATH = OUTPUT_DIR / "tfidf.npz"
TFIDF_FEATURES_PATH = OUTPUT_DIR / "tfidf_features.csv"
BOW_PATH = OUTPUT_DIR / "bow.npz"
BOW_FEATURES_PATH = OUTPUT_DIR / "bow_features.csv"
RESULTS_PATH = OUTPUT_DIR / "classification_results.csv"


st.set_page_config(
    page_title="StyleSense AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# NLTK RESOURCE SETUP
# ---------------------------------------------------------

@st.cache_resource
def ensure_nltk_resources():
    """Download and cache NLTK resources on the deployment server."""

    import nltk

    nltk_dir = str(Path("/tmp/nltk_data"))
    os.makedirs(nltk_dir, exist_ok=True)

    if nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_dir)

    resources = {
        "stopwords": "corpora/stopwords",
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "words": "corpora/words",
    }

    for package, resource_path in resources.items():

        try:
            nltk.data.find(resource_path)

        except LookupError:

            success = nltk.download(
                package,
                download_dir=nltk_dir,
                quiet=True,
            )

            if not success:
                raise RuntimeError(
                    f"Failed to download NLTK resource: {package}"
                )

            nltk.data.find(resource_path)

    return True


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

@st.cache_data
def load_dataset():

    if not DATA_PATH.exists():
        return None

    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_sparse_matrix(path_string):

    path = Path(path_string)

    if not path.exists():
        return None

    return load_npz(path)


@st.cache_data
def load_feature_names(path_string):

    path = Path(path_string)

    if not path.exists():
        return []

    return pd.read_csv(path)["Feature"].astype(str).tolist()


df = load_dataset()


# ---------------------------------------------------------
# SIDEBAR / NAVIGATION
# ---------------------------------------------------------

st.sidebar.title("🛍️ StyleSense AI")
st.sidebar.caption("Fashion Review Intelligence")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Review Analyzer",
        "ML Insights",
        "Topic Explorer",
    ],
)

st.sidebar.divider()
st.sidebar.caption("NLP • Machine Learning • E-commerce Analytics")


# ---------------------------------------------------------
# COMMON HELPERS
# ---------------------------------------------------------

def require_dataset():

    if df is None:

        st.error(
            "Processed dataset not found. Run src/numerical.py first. "
            "Expected file: outputs/processed_reviews.csv"
        )

        st.stop()


def show_metric(label, value, help_text=None):

    st.metric(label, value, help=help_text)


def get_recommendation_model():

    import joblib

    candidates = [
        MODEL_DIR / "best_model.joblib",
        MODEL_DIR / "linearsvc_pipeline.joblib",
        MODEL_DIR / "linear_svc_pipeline.joblib",
    ]

    for path in candidates:

        if path.exists():
            return joblib.load(path), path.name

    return None, None


# ---------------------------------------------------------
# REVIEW PREPROCESSING
# ---------------------------------------------------------

def preprocess_for_prediction(text):
    """
    Preprocess a customer review using the same approach
    as the numerical preprocessing pipeline.
    """

    import nltk

    from nltk.corpus import stopwords, words
    from nltk.tokenize import word_tokenize

    try:

        ensure_nltk_resources()

        stop_words = set(stopwords.words("english"))
        english_vocab = set(words.words())

        tokens = word_tokenize(text)

    except Exception as error:

        st.error(f"NLTK preprocessing failed: {error}")

        return ""

    tokens = [word.lower() for word in tokens]

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    tokens = [
        word for word in tokens
        if word.isalpha()
    ]

    tokens = [
        word for word in tokens
        if word in english_vocab and len(word) > 2
    ]

    return " ".join(tokens)


# ---------------------------------------------------------
# PAGE 1: OVERVIEW
# ---------------------------------------------------------

if page == "Overview":

    st.title("StyleSense AI")

    st.subheader("Fashion Review Analytics Dashboard")

    st.write(
        "Explore customer feedback from the Women's E-Commerce Clothing "
        "Reviews dataset using text mining and machine learning."
    )

    require_dataset()

    total_reviews = len(df)

    rating_col = "Rating" if "Rating" in df.columns else None

    target_col = (
        "Recommended IND"
        if "Recommended IND" in df.columns
        else None
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Reviews", f"{total_reviews:,}")

    if rating_col:

        c2.metric(
            "Average Rating",
            f"{df[rating_col].mean():.2f}/5"
        )

    if target_col:

        recommended_pct = (
            100
            * pd.to_numeric(
                df[target_col],
                errors="coerce"
            ).mean()
        )

        c3.metric(
            "Recommended",
            f"{recommended_pct:.1f}%"
        )

        c4.metric(
            "Not Recommended",
            f"{100 - recommended_pct:.1f}%"
        )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Rating Distribution")

        if rating_col:

            fig, ax = plt.subplots(figsize=(7, 4))

            sns.countplot(
                data=df,
                x=rating_col,
                order=sorted(df[rating_col].dropna().unique()),
                ax=ax,
                color="#4C78A8",
            )

            ax.set_xlabel("Rating")
            ax.set_ylabel("Number of reviews")

            st.pyplot(fig)

            plt.close(fig)

        else:

            st.info("Rating column not available.")

    with right:

        st.subheader("Recommendation Distribution")

        if target_col:

            counts = df[target_col].value_counts().sort_index()

            labels = {
                0: "Not Recommended",
                1: "Recommended",
            }

            fig, ax = plt.subplots(figsize=(7, 4))

            ax.pie(
                counts.values,
                labels=[
                    labels.get(int(x), str(x))
                    for x in counts.index
                ],
                autopct="%1.1f%%",
                startangle=90,
            )

            ax.axis("equal")

            st.pyplot(fig)

            plt.close(fig)

        else:

            st.info(
                "Recommendation target column not available."
            )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    info1, info2, info3 = st.columns(3)

    info1.metric("Columns", df.shape[1])

    info2.metric(
        "Missing Values",
        int(df.isna().sum().sum())
    )

    info3.metric(
        "Duplicate Rows",
        int(df.duplicated().sum())
    )

    if "Department Name" in df.columns:

        st.subheader("Reviews by Department")

        dept_counts = (
            df["Department Name"]
            .fillna("Unknown")
            .value_counts()
            .head(10)
        )

        st.bar_chart(dept_counts)


# ---------------------------------------------------------
# PAGE 2: REVIEW ANALYZER
# ---------------------------------------------------------

elif page == "Review Analyzer":

    st.title("Review Analyzer")

    st.write(
        "Enter a clothing review to predict whether the customer "
        "would recommend the product."
    )

    model, model_filename = get_recommendation_model()

    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.joblib"

    if model is None:

        st.warning(
            "A trained classification pipeline has not been saved yet. "
            "Save your trained LinearSVC pipeline in the models directory."
        )

    elif not vectorizer_path.exists():

        st.warning(
            "TF-IDF vectorizer not found. Save it from numerical.py as "
            "models/tfidf_vectorizer.joblib."
        )

    review = st.text_area(
        "Enter customer review",
        placeholder=(
            "Example: The fabric is soft and comfortable, "
            "and the fit is perfect."
        ),
        height=160,
    )

    if st.button("Analyze Review", type="primary"):

        if not review.strip():

            st.warning("Please enter a review first.")

        elif model is None or not vectorizer_path.exists():

            st.error(
                "Prediction is unavailable until the model "
                "and vectorizer are saved."
            )

        else:

            import joblib

            cleaned = preprocess_for_prediction(review)

            if not cleaned:

                st.warning(
                    "The review became empty after preprocessing. "
                    "Try a longer review with meaningful English words."
                )

            else:

                vectorizer = joblib.load(vectorizer_path)

                features = vectorizer.transform([cleaned])

                prediction = int(model.predict(features)[0])

                st.subheader("Prediction")

                if prediction == 1:

                    st.success(
                        "The model predicts: Recommended"
                    )

                else:

                    st.info(
                        "The model predicts: Not Recommended"
                    )

                st.caption(
                    f"Model used: {model_filename}"
                )

                st.write("**Cleaned review text:**")

                st.code(cleaned)

                if hasattr(model, "decision_function"):

                    try:

                        score = float(
                            np.ravel(
                                model.decision_function(features)
                            )[0]
                        )

                        st.write(
                            f"Decision score: `{score:.4f}`"
                        )

                        st.caption(
                            "This is a model decision score, "
                            "not a probability."
                        )

                    except Exception:

                        pass

    st.divider()

    st.caption(
        "Predictions are generated by your trained model and should be "
        "treated as estimates, not guarantees of customer behavior."
    )


# ---------------------------------------------------------
# PAGE 3: ML INSIGHTS
# ---------------------------------------------------------

elif page == "ML Insights":

    st.title("Machine Learning Insights")

    st.write(
        "Compare the classification models trained to predict "
        "product recommendations."
    )

    if not RESULTS_PATH.exists():

        st.warning(
            "classification_results.csv was not found. "
            "Run your updated src/classification.py first."
        )

    else:

        results = pd.read_csv(
            RESULTS_PATH,
            index_col=0
        )

        metric_columns = [
            col
            for col in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ]
            if col in results.columns
        ]

        if metric_columns:

            st.subheader("Test Set Performance")

            st.dataframe(
                results[metric_columns].style.format("{:.4f}"),
                use_container_width=True,
            )

            st.subheader("Model Comparison")

            st.bar_chart(
                results[metric_columns]
            )

            best_model = (
                results["F1 Score"].astype(float).idxmax()
                if "F1 Score" in results.columns
                else None
            )

            if best_model:

                st.success(
                    "Highest test-set F1 score in the saved results: "
                    f"{best_model}"
                )

        else:

            st.info(
                "No recognized evaluation metric columns were found."
            )

    matrix_path = (
        OUTPUT_DIR / "classification_confusion_matrices.png"
    )

    if matrix_path.exists():

        st.subheader("Confusion Matrices")

        st.image(
            str(matrix_path),
            use_container_width=True
        )

    feature_path = OUTPUT_DIR / "tfidf_features.csv"

    if feature_path.exists():

        st.subheader("TF-IDF Vocabulary")

        features = load_feature_names(
            str(feature_path)
        )

        st.write(
            f"Number of TF-IDF features: **{len(features):,}**"
        )

        st.caption(
            "The vocabulary is generated by the preprocessing "
            "and TF-IDF stage."
        )


# ---------------------------------------------------------
# PAGE 4: TOPIC EXPLORER
# ---------------------------------------------------------

elif page == "Topic Explorer":

    st.title("Topic Explorer")

    st.write(
        "Explore recurring themes in customer reviews using K-Means "
        "clustering and Latent Dirichlet Allocation (LDA)."
    )

    bow = load_sparse_matrix(str(BOW_PATH))

    bow_features = load_feature_names(
        str(BOW_FEATURES_PATH)
    )

    tfidf = load_sparse_matrix(str(TFIDF_PATH))

    tfidf_features = load_feature_names(
        str(TFIDF_FEATURES_PATH)
    )

    topic_tab, cluster_tab = st.tabs(
        [
            "LDA Topic Modeling",
            "K-Means Clustering"
        ]
    )

    # ---------------------------------------------
    # LDA TOPIC MODELING
    # ---------------------------------------------

    with topic_tab:

        if bow is None or not bow_features:

            st.warning(
                "BoW files are missing. Run src/numerical.py first."
            )

        else:

            n_topics = st.slider(
                "Number of topics",
                2,
                12,
                8
            )

            if st.button(
                "Discover Topics",
                key="discover_topics"
            ):

                with st.spinner(
                    "Fitting LDA model. This may take a little time..."
                ):

                    lda = LatentDirichletAllocation(
                        n_components=n_topics,
                        random_state=42,
                        max_iter=10,
                        learning_method="batch",
                    )

                    lda.fit(bow)

                st.subheader("Discovered Topics")

                for topic_idx, topic in enumerate(
                    lda.components_
                ):

                    top_indices = topic.argsort()[-10:][::-1]

                    topic_words = [
                        bow_features[i]
                        for i in top_indices
                    ]

                    with st.expander(
                        f"Topic {topic_idx + 1}"
                    ):

                        st.write(
                            ", ".join(topic_words)
                        )

                st.caption(
                    "Topics are machine-generated word groups. "
                    "Their interpretation requires reviewing "
                    "representative documents."
                )

    # ---------------------------------------------
    # K-MEANS CLUSTERING
    # ---------------------------------------------

    with cluster_tab:

        if tfidf is None or not tfidf_features:

            st.warning(
                "TF-IDF files are missing. Run src/numerical.py first."
            )

        else:

            k = st.slider(
                "Number of clusters",
                2,
                10,
                8
            )

            maximum_sample = min(
                10000,
                tfidf.shape[0]
            )

            minimum_sample = min(
                1000,
                maximum_sample
            )

            default_sample = min(
                5000,
                maximum_sample
            )

            sample_size = st.slider(
                "Documents to use",
                min_value=minimum_sample,
                max_value=maximum_sample,
                value=default_sample,
                step=1 if maximum_sample < 1500 else 500,
            )

            if st.button(
                "Run Clustering",
                key="run_clustering"
            ):

                with st.spinner(
                    "Clustering reviews..."
                ):

                    rng = np.random.RandomState(42)

                    indices = rng.choice(
                        tfidf.shape[0],
                        size=sample_size,
                        replace=False,
                    )

                    sample = tfidf[indices]

                    cluster_model = MiniBatchKMeans(
                        n_clusters=k,
                        random_state=42,
                        batch_size=512,
                        n_init=3,
                    )

                    labels = cluster_model.fit_predict(sample)

                    st.subheader("Cluster Distribution")

                    cluster_counts = (
                        pd.Series(labels)
                        .value_counts()
                        .sort_index()
                    )

                    st.bar_chart(cluster_counts)

                    st.subheader(
                        "Important Words per Cluster"
                    )

                    for cluster_id, center in enumerate(
                        cluster_model.cluster_centers_
                    ):

                        top_indices = np.argsort(
                            center
                        )[-10:][::-1]

                        top_words = [
                            tfidf_features[i]
                            for i in top_indices
                        ]

                        with st.expander(
                            f"Cluster {cluster_id} "
                            f"({cluster_counts.get(cluster_id, 0)} reviews)"
                        ):

                            st.write(
                                ", ".join(top_words)
                            )

                    st.caption(
                        "Clustering was performed on a reproducible "
                        f"random sample of {sample_size:,} reviews "
                        "to keep the dashboard responsive."
                    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.sidebar.divider()

st.sidebar.caption(
    "StyleSense AI • NLP & Machine Learning"
)
