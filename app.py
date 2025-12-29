import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

MODEL_PATH = "models/pipeline.joblib"

st.set_page_config(page_title="Spam Email Classifier", layout="wide")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

def proba_spam(model, text: str) -> float:
    # pipeline 有 predict_proba 才能拿機率（LogReg 有）
    proba = model.predict_proba([text])[0][1]
    return float(proba)

def label_from_proba(p: float, threshold: float) -> str:
    return "SPAM" if p >= threshold else "HAM"

model = load_model()

st.title("📧 Spam Email Classification Demo")

if model is None:
    st.error("找不到模型檔 models/pipeline.joblib。請先在本機執行：python train.py 產生模型後再跑 app。")
    st.stop()

with st.sidebar:
    st.header("設定")
    threshold = st.slider("判定為 Spam 的門檻 (threshold)", 0.1, 0.9, 0.5, 0.05)
    st.caption("門檻越低 → 越容易判為 Spam（召回率高，但誤判也可能增加）")

tab1, tab2, tab3 = st.tabs(["🔎 單封判斷", "📁 批次預測", "📊 模型資訊"])

# --- Tab 1: Single prediction ---
with tab1:
    col1, col2 = st.columns([1.2, 1])
    with col1:
        text = st.text_area("貼上 Email 內容", height=220, placeholder="Paste email text here...")
        btn = st.button("開始判斷", type="primary")
    with col2:
        if btn and text.strip():
            p = proba_spam(model, text)
            pred = label_from_proba(p, threshold)
            st.metric("預測結果", pred)
            st.metric("Spam 機率", f"{p:.3f}")
            st.progress(min(max(p, 0.0), 1.0))

            st.write("**解釋（簡版）**")
            st.write("- 這是以 TF-IDF 文字特徵 + Logistic Regression 訓練的分類器。")
        elif btn:
            st.warning("請先貼上內容。")

# --- Tab 2: Batch prediction ---
with tab2:
    st.write("上傳 CSV（需包含 `text` 欄位）。我會輸出預測與機率，可下載。")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        if "text" not in df.columns:
            st.error("CSV 必須包含 `text` 欄位。")
        else:
            texts = df["text"].astype(str).fillna("")
            probas = model.predict_proba(texts.tolist())[:, 1]
            preds = [label_from_proba(float(p), threshold) for p in probas]

            out = df.copy()
            out["spam_proba"] = probas
            out["prediction"] = preds

            st.dataframe(out.head(30), use_container_width=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "下載預測結果 CSV",
                data=csv_bytes,
                file_name="spam_predictions.csv",
                mime="text/csv"
            )

# --- Tab 3: Model info (optional demo metrics) ---
with tab3:
    st.write("如果你想在 Demo 顯示評估指標，建議在 `train.py` 訓練時把測試集預測結果存成檔案再讀入。")
    st.info("目前這頁先提供「Top 關鍵詞」展示（若模型是 Logistic Regression + TFIDF）。")

    # 嘗試抓出 TFIDF + LR 權重做特徵解釋
    try:
        tfidf = model.named_steps["tfidf"]
        clf = model.named_steps["clf"]
        feature_names = tfidf.get_feature_names_out()
        coefs = clf.coef_[0]

        top_n = 20
        top_spam_idx = coefs.argsort()[-top_n:][::-1]
        top_ham_idx = coefs.argsort()[:top_n]

        colA, colB = st.columns(2)
        with colA:
            st.subheader("🔺 Top Spam Indicative Terms")
            spam_terms = pd.DataFrame({
                "term": feature_names[top_spam_idx],
                "weight": coefs[top_spam_idx]
            })
            st.dataframe(spam_terms, use_container_width=True)

        with colB:
            st.subheader("🔻 Top Ham Indicative Terms")
            ham_terms = pd.DataFrame({
                "term": feature_names[top_ham_idx],
                "weight": coefs[top_ham_idx]
            })
            st.dataframe(ham_terms, use_container_width=True)

    except Exception as e:
        st.warning(f"無法顯示關鍵詞（可能你換了模型或 pipeline 不同）：{e}")
