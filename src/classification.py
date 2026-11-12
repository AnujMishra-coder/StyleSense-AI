
# Classification: Multinomial Naive Bayes, LinearSVC and KNN

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import load_npz

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    make_scorer
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def main():

    print("\n======= StyleSense AI: Classification =======")

    data_path = os.path.join(OUTPUT_DIR, "processed_reviews.csv")
    tfidf_path = os.path.join(OUTPUT_DIR, "tfidf.npz")
    feature_path = os.path.join(OUTPUT_DIR, "tfidf_features.csv")

    for path in [data_path, tfidf_path, feature_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file missing: {path}. "
                "Run src/numerical.py first."
            )

    # Load data
    print("Loading processed data...")

    df = pd.read_csv(data_path)
    X = load_npz(tfidf_path)

    feature_names = (
        pd.read_csv(feature_path)["Feature"]
        .astype(str)
        .tolist()
    )

    if len(df) != X.shape[0] or len(feature_names) != X.shape[1]:
        raise ValueError(
            "Dataset, TF-IDF matrix, and feature vocabulary "
            "dimensions do not match."
        )

    # Prepare target
    target = "Recommended IND"

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is missing.")

    valid = df[target].notna().to_numpy()

    df = df.loc[valid].reset_index(drop=True)
    X = X[valid]

    y = pd.to_numeric(
        df[target], errors="raise"
    ).astype(int).to_numpy()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    # Chi-square feature analysis
    k = min(1000, X_train.shape[1])

    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(X_train, y_train)

    scores = np.nan_to_num(
        selector.scores_,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    top_idx = np.argsort(scores)[::-1][:20]

    pd.DataFrame({
        "Feature": [feature_names[i] for i in top_idx],
        "Chi2_Score": scores[top_idx]
    }).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "top_classification_features.csv"
        ),
        index=False
    )

    # Models
    models = {
        "Naive Bayes": MultinomialNB(),

        "LinearSVC": LinearSVC(
            max_iter=10000,
            random_state=42
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=1
        )
    }

    # Resource-efficient cross-validation
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",

        "precision": make_scorer(
            precision_score,
            zero_division=0
        ),

        "recall": make_scorer(
            recall_score,
            zero_division=0
        ),

        "f1": make_scorer(
            f1_score,
            zero_division=0
        )
    }

    rows = []
    predictions_by_model = {}

    best_name = None
    best_f1 = -1.0

    # Train and evaluate each model
    for name, estimator in models.items():

        print(f"\nTraining/evaluating {name}...")

        pipeline = Pipeline([
            ("feature_selection", SelectKBest(chi2, k=k)),
            ("classifier", estimator)
        ])

        # Run CV sequentially to avoid Windows resource exhaustion
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise"
        )

        print(f"{name} cross-validation completed.")

        # Fit final model
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        predictions_by_model[name] = predictions

        # Calculate metrics
        row = {
            "Model": name,

            "CV Accuracy": cv_scores["test_accuracy"].mean(),
            "CV Precision": cv_scores["test_precision"].mean(),
            "CV Recall": cv_scores["test_recall"].mean(),
            "CV F1 Score": cv_scores["test_f1"].mean(),

            "Accuracy": accuracy_score(y_test, predictions),

            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),

            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),

            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            )
        }

        rows.append(row)

        # Save model
        filename = (
            name.lower().replace(" ", "_")
            + "_pipeline.joblib"
        )

        joblib.dump(
            pipeline,
            os.path.join(MODEL_DIR, filename)
        )

        # Classification report
        print(f"\n{name} test-set report:")

        print(
            classification_report(
                y_test,
                predictions,
                target_names=[
                    "Not Recommended",
                    "Recommended"
                ],
                zero_division=0
            )
        )

        # Identify best model
        if row["F1 Score"] > best_f1:

            best_f1 = row["F1 Score"]
            best_name = name

    # Model comparison
    results = pd.DataFrame(rows).set_index("Model")

    results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "classification_results.csv"
        )
    )

    print("\nModel comparison:")
    print(results.round(4))

    print(f"\nBest model by test F1: {best_name}")

    # Save best model
    best_filename = (
        best_name.lower().replace(" ", "_")
        + "_pipeline.joblib"
    )

    best_model = joblib.load(
        os.path.join(MODEL_DIR, best_filename)
    )

    joblib.dump(
        best_model,
        os.path.join(MODEL_DIR, "best_model.joblib")
    )

    # Confusion matrices
    fig, axes = plt.subplots(
        1, 3,
        figsize=(17, 5)
    )

    for ax, (name, predictions) in zip(
        axes,
        predictions_by_model.items()
    ):

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[
                "Not Recommended",
                "Recommended"
            ],
            yticklabels=[
                "Not Recommended",
                "Recommended"
            ],
            ax=ax
        )

        ax.set_title(f"{name} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()

    plot_path = os.path.join(
        OUTPUT_DIR,
        "classification_confusion_matrices.png"
    )

    plt.savefig(
        plot_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nClassification completed successfully.")
    print("Models and evaluation outputs saved.")


if __name__ == "__main__":
    main()
