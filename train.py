import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = "data/spam.csv"
MODEL_PATH = "models/pipeline.joblib"

def normalize_label(s: str) -> int:
    s = str(s).strip().lower()
    if s in ["spam", "1", "true", "yes"]:
        return 1
    if s in ["ham", "0", "false", "no"]:
        return 0
    # 如果你的資料是其他標籤，自己在這裡補
    raise ValueError(f"Unknown label: {s}")

def main():
    df = pd.read_csv(DATA_PATH)

    # 你可以在這裡對應自己的欄位名
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column (spam/ham or 1/0).")

    df = df.dropna(subset=["text", "label"]).copy()
    y = df["label"].apply(normalize_label)
    X = df["text"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2
            )),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            ))
        ]
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
