# Text preprocessing and document representation: TF-IDF + Bag of Words

import os
import joblib
import pandas as pd
import nltk

from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import save_npz

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Womens Clothing E-Commerce Reviews.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def ensure_nltk_resources():
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/words", "words"),
    ]
    for resource_path, package in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package, quiet=True)

def preprocess_text(text, stop_words, english_vocab):
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [token for token in tokens if token.isalpha()]
    tokens = [
        token for token in tokens
        if len(token) > 2 and token in english_vocab
    ]
    return " ".join(tokens)

def main():
    print("\n======= StyleSense AI: Numerical Processing =======")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place 'Womens Clothing E-Commerce Reviews.csv' in the project root."
        )

    ensure_nltk_resources()
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)

    review_column = "Review Text"
    if review_column not in df.columns:
        raise ValueError(f"Required column '{review_column}' not found.")

    stop_words = set(stopwords.words("english"))
    english_vocab = set(words.words())

    print("Preprocessing review text...")
    df["CLEANED_TEXT"] = df[review_column].fillna("").apply(
        lambda text: preprocess_text(text, stop_words, english_vocab)
    )
    df.drop(columns=[review_column], inplace=True)

    processed_path = os.path.join(OUTPUT_DIR, "processed_reviews.csv")
    df.to_csv(processed_path, index=False)
    print("Saved:", processed_path)

    print("Building TF-IDF matrix...")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df["CLEANED_TEXT"])

    save_npz(os.path.join(OUTPUT_DIR, "tfidf.npz"), tfidf_matrix)
    pd.DataFrame({
        "Feature": tfidf_vectorizer.get_feature_names_out()
    }).to_csv(os.path.join(OUTPUT_DIR, "tfidf_features.csv"), index=False)

    joblib.dump(
        tfidf_vectorizer,
        os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
    )
    print("TF-IDF shape:", tfidf_matrix.shape)

    print("Building Bag-of-Words matrix...")
    bow_vectorizer = CountVectorizer(
        max_features=10000,
        max_df=0.85,
        min_df=2
    )
    bow_matrix = bow_vectorizer.fit_transform(df["CLEANED_TEXT"])

    save_npz(os.path.join(OUTPUT_DIR, "bow.npz"), bow_matrix)
    pd.DataFrame({
        "Feature": bow_vectorizer.get_feature_names_out()
    }).to_csv(os.path.join(OUTPUT_DIR, "bow_features.csv"), index=False)
    joblib.dump(
        bow_vectorizer,
        os.path.join(MODEL_DIR, "bow_vectorizer.joblib")
    )
    print("BoW shape:", bow_matrix.shape)
    print("\nNumerical processing completed.")

if __name__ == "__main__":
    main()
