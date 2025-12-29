import io
import zipfile
import requests
import joblib
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

st.set_page_config(page_title="Spam Email Classifier", layout="wide")

DATA_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

@st.cache_data(show_spinner=False)
def load_dataset():
    """
    Download & parse SMS Spam Collection dataset from UCI (zip).
    Returns DataFrame with columns: label, text
    """
    r = requests.get(DATA_ZIP_URL, timeout=30)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        # zip 裡面通常是 'SMSSpamCollection'
        name = [n for n in z.namelist() if "SMSSpamCollection" in n][0]
        raw = z.read(name).decode("utf-8", errors="ignore")

    # 每行：label\ttext
    rows = []
    for line in raw.splitlines():
        if "\t" in line:
            lab, txt = line.split("\t", 1)
            rows.append((lab.strip(), txt.strip()))
    df = pd.DataFrame(rows, columns=["label", "text"])
    return df

@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    """
    Train TF-IDF + LogisticRegression pipeline and return:
    model, metrics dict, confusion matrix
    """
    y = df["label"].map(lambda x: 1 if str(x).lower().strip() == "spam" else 0)
    X = df["text"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
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
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "test_size": int(len(X_test)),
    }

    return pipe, metrics, cm

def predict_one(model, text: str, threshold: float):
    proba = float(model.predict_proba([text])[0][1])
    pred = "SPAM" if proba >= threshold else "HAM"
    return pred, proba

st.title("📧 Spam Email Classification (Streamlit Demo)")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Spam threshold", 0.10, 0.90, 0.50, 0.05)
    st.caption("threshold 越低 → 越容易判為 spam")

with st.spinner("Loading dataset..."):
    df = load_dataset()

with st.spinner("Training model (cached)..."):
    model, metrics, cm = train_model(df)

tab1, tab2, tab3 = st.tabs(["🔎 Single Email", "📁 Batch Predict", "📊 Model Metrics"])

# --- Tab 1: Single ---
with tab1:
    col1, col2 = st.columns([1.2, 1])
    with col1:
        text = st.text_area("Paste email content", height=220, placeholder="Paste email text here...")
        run = st.button("Predict", type="primary")
    with col2:
        if run:
            if not text.strip():
                st.warning("請先貼上 email 內容")
            else:
                pred, proba = predict_one(model, text, threshold)
                st.metric("Prediction", pred)
                st.metric("Spam probability", f"{proba:.3f}")
                st.progress(min(max(proba, 0.0), 1.0))

# --- Tab 2: Batch ---
with tab2:
    st.write("上傳 CSV（需包含 `text` 欄位），會回傳 prediction 與 spam_proba，可下載。")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        bdf = pd.read_csv(up)
        if "text" not in bdf.columns:
            st.error("你的 CSV 需要有 `text` 欄位（每列是一封 email）")
        else:
            texts = bdf["text"].astype(str).fillna("").tolist()
            probas = model.predict_proba(texts)[:, 1]
            preds = ["SPAM" if float(p) >= threshold else "HAM" for p in probas]

            out = bdf.copy()
            out["spam_proba"] = probas
            out["prediction"] = preds

            st.dataframe(out.head(30), use_container_width=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download results CSV",
                data=csv_bytes,
                file_name="spam_predictions.csv",
                mime="text/csv"
            )

# --- Tab 3: Metrics ---
with tab3:
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    colB.metric("Precision", f"{metrics['precision']:.3f}")
    colC.metric("Recall", f"{metrics['recall']:.3f}")
    colD.metric("F1", f"{metrics['f1']:.3f}")

    st.write(f"Test set size: **{metrics['test_size']}**")

    st.subheader("Confusion Matrix")
    st.write("Rows = True label (ham=0, spam=1), Cols = Pred label")
    st.table(pd.DataFrame(cm, index=["true_ham(0)", "true_spam(1)"], columns=["pred_ham(0)", "pred_spam(1)"]))

    st.subheader("Top indicative terms (LogReg weights)")
    try:
        tfidf = model.named_steps["tfidf"]
        clf = model.named_steps["clf"]
        feat = tfidf.get_feature_names_out()
        w = clf.coef_[0]

        top_n = 20
        top_spam = w.argsort()[-top_n:][::-1]
        top_ham = w.argsort()[:top_n]

        c1, c2 = st.columns(2)
        with c1:
            st.write("🔺 Spam-indicative")
            st.dataframe(pd.DataFrame({"term": feat[top_spam], "weight": w[top_spam]}), use_container_width=True)
        with c2:
            st.write("🔻 Ham-indicative")
            st.dataframe(pd.DataFrame({"term": feat[top_ham], "weight": w[top_ham]}), use_container_width=True)
    except Exception as e:
        st.warning(f"無法顯示關鍵詞：{e}")
