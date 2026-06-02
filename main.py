import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import os
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- IMPORT PIPELINE TIỀN XỬ LÝ (ĐÃ TÍCH HỢP PYVI) ---
from utils.preprocessing import preprocess_pipeline

# ==============================================================================
# CẤU HÌNH GIAO DIỆN VÀ ĐƯỜNG DẪN MÔ HÌNH
# ==============================================================================
st.set_page_config(page_title="VietText Analyzer Dashboard", page_icon="🚀", layout="wide")

# --- ĐƯỜNG DẪN MÔ HÌNH PHOBERT ---
MODEL_PHOBERT         = "./models/phobert"
LABEL_ENCODER_PHOBERT = "./models/phobert/label_encoder.pkl"

# --- ĐƯỜNG DẪN MÔ HÌNH TF-IDF (Phân tách theo cấu trúc GitHub mới của bạn) ---
MODEL_TFIDF_SENTIMENT   = "./models/tfidf/sentiment/baseline_sentiment_model.pkl"
LE_TFIDF_SENTIMENT      = "./models/tfidf/sentiment/baseline_sentiment_label_encoder.pkl"

MODEL_TFIDF_TOPIC       = "./models/tfidf/topic/baseline_topic_model.pkl"
LE_TFIDF_TOPIC          = "./models/tfidf/topic/baseline_topic_label_encoder.pkl"

DATASET_EXCEL = "./dataset/train.xlsx"
VALID_PATH    = "./dataset/validation.xlsx"

def _require_file(path: str, label: str):
    """Dừng app ngay nếu file bắt buộc không tồn tại."""
    if not os.path.exists(path):
        st.error(f"❌ Không tìm thấy file bắt buộc **{label}**:\n`{path}`\n\nVui lòng kiểm tra lại cấu trúc GitHub.")
        st.stop()

# ==============================================================================
# TẢI TOÀN BỘ MÔ HÌNH VÀO BỘ NHỚ ĐỆM (CACHE)
# ==============================================================================
@st.cache_resource
def load_all_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. PhoBERT ─────────────────────────────────────────────────────────────
    _require_file(os.path.join(MODEL_PHOBERT, "model.safetensors"), "PhoBERT weights")
    _require_file(LABEL_ENCODER_PHOBERT, "PhoBERT label encoder")

    try:
        phobert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PHOBERT)
        phobert_model.to(device)
        phobert_model.eval()
        phobert_tokenizer = AutoTokenizer.from_pretrained(MODEL_PHOBERT)
    except Exception as e:
        st.error(f"Lỗi khi tải PhoBERT: {e}")
        st.stop()

    with open(LABEL_ENCODER_PHOBERT, "rb") as f:
        phobert_le = pickle.load(f)

    # ── 2. TF-IDF Sentiment ───────────────────────────────────────────────────
    _require_file(MODEL_TFIDF_SENTIMENT, "TF-IDF Sentiment model")
    _require_file(LE_TFIDF_SENTIMENT, "TF-IDF Sentiment label encoder")
    try:
        with open(MODEL_TFIDF_SENTIMENT, "rb") as f:
            tfidf_sent_model = pickle.load(f)
        with open(LE_TFIDF_SENTIMENT, "rb") as f:
            tfidf_sent_le = pickle.load(f)
    except Exception as e:
        st.error(f"Lỗi khi tải TF-IDF Sentiment: {e}")
        st.stop()

    # ── 3. TF-IDF Topic ───────────────────────────────────────────────────────
    _require_file(MODEL_TFIDF_TOPIC, "TF-IDF Topic model")
    _require_file(LE_TFIDF_TOPIC, "TF-IDF Topic label encoder")
    try:
        with open(MODEL_TFIDF_TOPIC, "rb") as f:
            tfidf_topic_model = pickle.load(f)
        with open(LE_TFIDF_TOPIC, "rb") as f:
            tfidf_topic_le = pickle.load(f)
    except Exception as e:
        st.error(f"Lỗi khi tải TF-IDF Topic: {e}")
        st.stop()

    return (
        phobert_model, phobert_tokenizer, phobert_le,
        tfidf_sent_model, tfidf_sent_le,
        tfidf_topic_model, tfidf_topic_le,
        device
    )

(phobert_m, phobert_t, phobert_le,
 tfidf_s_m, tfidf_s_le,
 tfidf_t_m, tfidf_t_le,
 DEVICE) = load_all_models()

# ==============================================================================
# HÀM HỖ TRỢ XỬ LÝ DỰ ĐOÁN PHOBERT
# ==============================================================================
@torch.no_grad()
def phobert_predict_batch(texts, batch_size=32):
    all_labels = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i: i + batch_size]
        inputs = phobert_t(chunk, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        logits = phobert_m(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
        labels = phobert_le.inverse_transform(pred_ids)
        all_labels.extend(labels)
    return all_labels

# ==============================================================================
# CHUẨN HOÁ & HIỂN THỊ NHÃN CẢM XÚC
# ==============================================================================
def standardize_labels(label_list):
    out = []
    for l in label_list:
        s = str(l).strip().lower()
        if any(x in s for x in ["tiêu cực", "neg", "negative"]) or s in ["0", "0.0"]:
            out.append("0")
        elif any(x in s for x in ["trung lập", "neu", "neutral"]) or s in ["1", "1.0"]:
            out.append("1")
        elif any(x in s for x in ["tích cực", "pos", "positive"]) or s in ["2", "2.0"]:
            out.append("2")
        else:
            out.append(s)
    return out

def label_to_display(label_text):
    t = str(label_text).upper()
    if any(w in t for w in ["POS", "TÍCH CỰC", "2", "POSITIVE"]):
        st.success("🎯 TÍCH CỰC 😍")
    elif any(w in t for w in ["NEG", "TIÊU CỰC", "0", "NEGATIVE"]):
        st.error("🎯 TIÊU CỰC 😡")
    else:
        st.warning("🎯 TRUNG LẬP 😐")

# ==============================================================================
# ĐIỀU HƯỚNG SIDEBAR
# ==============================================================================
st.sidebar.title("🎮 Hệ Thống Điều Khiển")
st.sidebar.markdown("Chọn tính năng hiển thị đồ án:")
page = st.sidebar.radio("Danh mục trang:", [
    "🏠 Giới thiệu dự án & Dataset",
    "⚡ Trình dự đoán song song tổng lực",
    "📊 Chỉ số thực nghiệm & Ma trận nhầm lẫn",
])

# ==============================================================================
# TRANG 1: GIỚI THIỆU & THỐNG KÊ DATASET
# ==============================================================================
if page == "🏠 Giới thiệu dự án & Dataset":
    st.title("🔮 VietText Analyzer — NLP Research Dashboard")
    st.markdown("### Phân tích Sắc thái và Chủ đề Ý kiến Sinh viên bằng Machine Learning & Deep Learning")
    st.divider()

    st.header("1. Giới thiệu đề tài")
    st.write("""
    Hệ thống tích hợp đa mô hình phục vụ đánh giá so sánh hiệu năng toán học:
    - **TF-IDF + Machine Learning:** Tốc độ siêu tốc, tối ưu tài nguyên phần cứng, hoạt động dựa trên tần suất từ vựng độc lập.
    - **PhoBERT Transformer:** Kiến trúc ngôn ngữ lớn tiên tiến (SOTA) tối ưu chuyên biệt cho tiếng Việt, nắm bắt tốt ngữ cảnh đảo chiều.
    """)

    st.header("2. Khám phá Bộ dữ liệu (Dataset Explorer)")

    @st.cache_data
    def load_dataset():
        _require_file(DATASET_EXCEL, "Dataset train.xlsx")
        df = pd.read_excel(DATASET_EXCEL)
        sent_map  = {0: "Tiêu cực", 1: "Trung lập", 2: "Tích cực"}
        topic_map = {0: "Cơ sở vật chất", 1: "Chương trình đào tạo", 2: "Giảng viên", 3: "Học phí & Khác"}
        if len(df.columns) >= 1: df["sentence"]       = df.iloc[:, 0]
        if len(df.columns) >= 2: df["sentiment_label"] = df.iloc[:, 1].map(sent_map).fillna(df.iloc[:, 1])
        if len(df.columns) >= 3: df["topic_label"]    = df.iloc[:, 2].map(topic_map).fillna
