"""
Task 3: Logistic Regression with GloVe embeddings (teacher workflow).

Data layout (same as course materials):
  ./datasets/imdb_top_500.csv  (or ./imdb_top_500.csv in project root)
  ./datasets/tiny_glove.json   (or ./tiny_glove.json in project root)

If tiny_glove.json is missing, place Stanford `glove.6B.50d.txt` under ./datasets/
and this script will load only the words that appear in the IMDB texts (no gensim).

Install dependencies (Tsinghua mirror):

    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \\
        --trusted-host pypi.tuna.tsinghua.edu.cn

Or run: powershell -ExecutionPolicy Bypass -File .\\install_deps.ps1
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
CSV_PATH = DATASETS / "imdb_top_500.csv"
GLOVE_JSON = DATASETS / "tiny_glove.json"
GLOVE_JSON_ROOT = ROOT / "tiny_glove.json"
GLOVE_TXT = DATASETS / "glove.6B.50d.txt"


def tokenize(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()


def collect_vocab(texts: np.ndarray) -> set[str]:
    vocab: set[str] = set()
    for t in texts:
        vocab.update(tokenize(t))
    return vocab


def load_glove_from_txt(path: Path, vocab: set[str]) -> dict[str, list[float]]:
    glove: dict[str, list[float]] = {}
    needed = set(vocab)
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 51:
                continue
            w = parts[0]
            if w in needed:
                glove[w] = [float(x) for x in parts[1:]]
                needed.discard(w)
                if not needed:
                    break
    return glove


def load_glove(texts: pd.Series) -> dict[str, list[float]]:
    for path in (GLOVE_JSON, GLOVE_JSON_ROOT):
        if path.is_file():
            print("Using GloVe JSON:", path)
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    vocab = collect_vocab(texts.values)
    if not GLOVE_TXT.is_file():
        raise FileNotFoundError(
            f"Missing tiny_glove.json (expected {GLOVE_JSON} or {GLOVE_JSON_ROOT}) "
            f"and {GLOVE_TXT}. "
            "Add the course tiny_glove.json, or download Stanford GloVe 6B 50d "
            "(`glove.6B.50d.txt`) into ./datasets/."
        )
    print(f"Loading filtered GloVe from {GLOVE_TXT.name} (first run may take a minute)...")
    glove = load_glove_from_txt(GLOVE_TXT, vocab)
    print("GloVe hits for corpus vocab:", len(glove), "/", len(vocab))
    return glove


def get_embedding(text: str, glove: dict[str, list[float]], dim: int = 50) -> np.ndarray:
    tokens = tokenize(text)
    vectors = [np.array(glove[word]) for word in tokens if word in glove]
    if len(vectors) == 0:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)


def main() -> None:
    df = None
    for p in (CSV_PATH, ROOT / "imdb_top_500.csv"):
        if p.is_file():
            df = pd.read_csv(p)
            print("Using CSV:", p)
            break
    if df is None:
        raise FileNotFoundError(f"Missing {CSV_PATH} or {ROOT / 'imdb_top_500.csv'}")

    glove = load_glove(df["text"])

    print("Dataset size:", len(df))
    print("Columns:", df.columns.tolist())
    print("\nFirst review preview:")
    print(df["text"].iloc[0][:300])
    print("\nFirst label:", df["label"].iloc[0])
    print("First rating:", df["rating"].iloc[0])
    print("\nVocabulary size (glove keys):", len(glove))

    sample_tokens = tokenize(df["text"].iloc[0])
    print("\nFirst 20 tokens:")
    print(sample_tokens[:20])
    print("Total token count:", len(sample_tokens))

    sample_vector = get_embedding(df["text"].iloc[0], glove)
    print("\nEmbedding shape:", sample_vector.shape)
    print("First 10 embedding values:")
    print(sample_vector[:10])

    X = np.array([get_embedding(text, glove) for text in df["text"]])
    y = df["label"].values
    texts = df["text"].values

    print("\nFeature matrix shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Texts shape:", texts.shape)
    print("Positive ratio:", np.mean(y))

    X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
        X, y, texts, test_size=0.2, random_state=42
    )
    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("\nFirst test review preview:")
    print(text_test[0][:300])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("\nFeature scaling complete.")
    print("First standardized training vector:")
    print(X_train[0][:10])

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("\nModel training complete.")

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    print("\nFirst 30 test predictions:")
    print(test_pred[:30])
    print("\nFirst 30 true labels:")
    print(y_test[:30])

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    print("\nTrain Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)

    print("\n--- Step 11: sample test indices 85-89 ---")
    for i in range(85, 90):
        print(f"\nReview {i + 1}")
        print("-" * 60)
        print(text_test[i][:400])
        print("\nTrue Label:", y_test[i])
        print("Predicted Label:", test_pred[i])
        if y_test[i] == test_pred[i]:
            print("Result: Correct")
        else:
            print("Result: Incorrect")

    sample_reviews = [
        "This movie was fantastic with brilliant acting",
        "I hated this movie it was boring and terrible",
        "The film was okay not great but not bad",
    ]
    sample_X = np.array([get_embedding(text, glove) for text in sample_reviews])
    sample_X = scaler.transform(sample_X)
    sample_preds = model.predict(sample_X)
    print("\n--- Step 12: new reviews ---")
    for review, pred in zip(sample_reviews, sample_preds):
        print("\nReview:")
        print(review)
        print(
            "Predicted Sentiment:",
            "Positive" if pred == 1 else "Negative",
        )


if __name__ == "__main__":
    main()
