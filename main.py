import re
import os
import time
import pickle
import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from utils.preprocessing import preprocess_pipeline

# ==============================================================================
# CẤU HÌNH GIAO DIỆN VÀ ĐƯỜNG DẪN MÔ HÌNH
# ==============================================================================
st.set_page_config(page_title="VietText Analyzer Dashboard", page_icon="🚀", layout="wide")

# --- ĐƯỜNG DẪN PHOBERT ---
MODEL_PHOBERT               = "./models/phobert"
LABEL_ENCODER_PHOBERT       = "./models/phobert/label_encoder.pkl"

MODEL_PHOBERT_TOPIC         = "./models/phobert_topic"
LABEL_ENCODER_PHOBERT_TOPIC = "./models/phobert_topic/label_encoder.pkl"

# --- ĐƯỜNG DẪN TF-IDF ---
MODEL_TFIDF_SENT            = "./models/tfidf/sentiment/baseline_sentiment_model.pkl"
LABEL_ENCODER_TFIDF_SENT    = "./models/tfidf/sentiment/baseline_sentiment_label_encoder.pkl"

MODEL_TFIDF_TOPIC           = "./models/tfidf/topic/baseline_topic_model.pkl"
LABEL_ENCODER_TFIDF_TOPIC   = "./models/tfidf/topic/baseline_topic_label_encoder.pkl"

DATASET_EXCEL = "./dataset/train.xlsx"
VALID_PATH    = "./dataset/validation.xlsx"

TOPIC_MAP_VI = {
    "curriculum": "Chương trình đào tạo 📚",
    "facility":   "Cơ sở vật chất 🏫",
    "lecturer":   "Giảng viên 👨‍🏫",
    "other":      "Khác / Tổng quan 📝",
}

def topic_vi(label: str) -> str:
    return TOPIC_MAP_VI.get(str(label).strip().lower(), label)

def _require_file(path: str, label: str):
    if not os.path.exists(path):
        st.error(f"❌ Không tìm thấy file bắt buộc {label}:\n`{path}`")
        st.stop()

# ==============================================================================
# TẢI MÔ HÌNH VÀO CACHE RESOURCE
# ==============================================================================
@st.cache_resource
def load_all_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── TF-IDF Sentiment ───────────────────────────────────────────────────────
    _require_file(MODEL_TFIDF_SENT,         "TF-IDF Sentiment model")
    _require_file(LABEL_ENCODER_TFIDF_SENT, "TF-IDF Sentiment label encoder")
    with open(MODEL_TFIDF_SENT, "rb") as f:
        tfidf_sent_m = pickle.load(f)
    with open(LABEL_ENCODER_TFIDF_SENT, "rb") as f:
        tfidf_sent_le = pickle.load(f)

    # ── TF-IDF Topic ──────────────────────────────────────────────────────────
    _require_file(MODEL_TFIDF_TOPIC,         "TF-IDF Topic model")
    _require_file(LABEL_ENCODER_TFIDF_TOPIC, "TF-IDF Topic label encoder")
    with open(MODEL_TFIDF_TOPIC, "rb") as f:
        tfidf_topic_m = pickle.load(f)
    with open(LABEL_ENCODER_TFIDF_TOPIC, "rb") as f:
        tfidf_topic_le = pickle.load(f)

    # ── PhoBERT Sentiment ─────────────────────────────────────────────────────
    _require_file(os.path.join(MODEL_PHOBERT, "model.safetensors"), "PhoBERT Sentiment weights")
    _require_file(LABEL_ENCODER_PHOBERT, "PhoBERT Sentiment label encoder")
    try:
        phobert_sent_m = AutoModelForSequenceClassification.from_pretrained(MODEL_PHOBERT)
        phobert_sent_m.to(device).eval()
        phobert_sent_t = AutoTokenizer.from_pretrained(MODEL_PHOBERT)
    except Exception as e:
        st.error(f"❌ Lỗi load PhoBERT Sentiment: {e}")
        st.stop()
    with open(LABEL_ENCODER_PHOBERT, "rb") as f:
        phobert_sent_le = pickle.load(f)

    # ── PhoBERT Topic (Tự động kích hoạt nếu tồn tại thư mục weights) ─────────
    phobert_topic_m = phobert_topic_t = phobert_topic_le = None
    if os.path.exists(os.path.join(MODEL_PHOBERT_TOPIC, "model.safetensors")):
        try:
            phobert_topic_m = AutoModelForSequenceClassification.from_pretrained(MODEL_PHOBERT_TOPIC)
            phobert_topic_m.to(device).eval()
            phobert_topic_t = AutoTokenizer.from_pretrained(MODEL_PHOBERT_TOPIC)
            with open(LABEL_ENCODER_PHOBERT_TOPIC, "rb") as f:
                phobert_topic_le = pickle.load(f)
        except Exception as e:
            st.warning(f"⚠️ PhoBERT Topic chưa load được (Sử dụng dự phòng bằng TF-IDF): {e}")
            phobert_topic_m = phobert_topic_t = phobert_topic_le = None

    return (
        tfidf_sent_m,  tfidf_sent_le,
        tfidf_topic_m, tfidf_topic_le,
        phobert_sent_m, phobert_sent_t, phobert_sent_le,
        phobert_topic_m, phobert_topic_t, phobert_topic_le,
        device,
    )

(tfidf_sent_m,  tfidf_sent_le,
 tfidf_topic_m, tfidf_topic_le,
 phobert_sent_m, phobert_sent_t, phobert_sent_le,
 phobert_topic_m, phobert_topic_t, phobert_topic_le,
 DEVICE) = load_all_models()

# ==============================================================================
# HÀM TRỢ GIÚP DỰ ĐOÁN (HELPER PREDICT)
# ==============================================================================
@torch.no_grad()
def phobert_predict_batch(texts, model, tokenizer, le, batch_size=32):
    all_labels = []
    for i in range(0, len(texts), batch_size):
        chunk  = texts[i: i + batch_size]
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        logits = model(**inputs).logits
        ids    = torch.argmax(logits, dim=-1).cpu().numpy()
        all_labels.extend(le.inverse_transform(ids))
    return all_labels

def tfidf_predict(texts, model, le):
    codes = model.predict(texts)
    return le.inverse_transform(codes)

# --- Dự đoán đơn kèm mảng xác suất chi tiết cho Trang 2 ---
@torch.no_grad()
def phobert_predict_single_with_prob(text, model, tokenizer, le):
    inputs = tokenizer([text], return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    logits = model(**inputs).logits
    probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0] # Mảng xác suất các lớp
    
    max_idx = np.argmax(probs)
    label = le.inverse_transform([max_idx])[0]
    
    # Tạo dict ánh xạ: tên nhãn gốc -> xác suất
    prob_dict = {str(le.classes_[idx]): float(probs[idx]) for idx in range(len(le.classes_))}
    return label, prob_dict

def tfidf_predict_single_with_prob(text, model, le):
    prob_matrix = model.predict_proba([text])[0]
    max_idx = np.argmax(prob_matrix)
    label = le.inverse_transform([max_idx])[0]
    
    prob_dict = {str(le.classes_[idx]): float(prob_matrix[idx]) for idx in range(len(le.classes_))}
    return label, prob_dict

# ==============================================================================
# CHUẨN HOÁ NHÃN SENTIMENT & HIỂN THỊ CHI TIẾT
# ==============================================================================
def standardize_sentiment(label_list):
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

def label_to_display_with_details(label_text, prob_dict):
    """Hiển thị nhãn chiến thắng kèm bảng phân rã xác suất chi tiết Tiêu cực/Trung lập/Tích cực"""
    t = str(label_text).upper()
    if any(w in t for w in ["POS", "TÍCH CỰC", "2", "POSITIVE"]):
        st.success("🎯 **SẮC THÁI CHÍNH:** TÍCH CỰC 😍")
    elif any(w in t for w in ["NEG", "TIÊU CỰC", "0", "NEGATIVE"]):
        st.error("🎯 **SẮC THÁI CHÍNH:** TIÊU CỰC 😡")
    else:
        st.warning("🎯 **SẮC THÁI CHÍNH:** TRUNG LẬP 😐")
        
    # Chuẩn hóa key của prob_dict về dạng hiển thị thân thiện
    display_probs = {"Tiêu cực 😡": 0.0, "Trung lập 😐": 0.0, "Tích cực 😍": 0.0}
    for k, v in prob_dict.items():
        sk = str(k).strip().lower()
        if any(x in sk for x in ["tiêu cực", "neg", "negative"]) or sk in ["0", "0.0"]:
            display_probs["Tiêu cực 😡"] = v
        elif any(x in sk for x in ["trung lập", "neu", "neutral"]) or sk in ["1", "1.0"]:
            display_probs["Trung lập 😐"] = v
        elif any(x in sk for x in ["tích cực", "pos", "positive"]) or sk in ["2", "2.0"]:
            display_probs["Tích cực 😍"] = v
        else:
            display_probs[k] = v

    # Hiển thị thanh tiến trình xác suất trực quan
    st.markdown("**📊 Phân bổ xác suất chi tiết:**")
    for name, score in display_probs.items():
        st.write(f"{name}: **{score * 100:.2f}%**")
        st.progress(int(score * 100))

def topic_to_display_with_details(label_text, prob_dict):
    """Hiển thị nhãn chủ đề kèm bảng phân rã xác suất chi tiết cho từng Topic"""
    st.info(f"📌 **CHỦ ĐỀ CHÍNH:** {topic_vi(label_text)}")
    
    st.markdown("**📊 Phân bổ xác suất chi tiết:**")
    for k, v in prob_dict.items():
        st.write(f"• {topic_vi(k)}: **{v * 100:.2f}%**")
        st.progress(int(v * 100))

# ==============================================================================
# SideBar Điều hướng nâng cao
# ==============================================================================
with st.sidebar:
    st.title("🎮 Hệ Thống Điều Khiển")
    st.markdown("---")
    page = st.radio(
        "Danh mục trang hệ thống:", 
        [
            "🏠 Giới thiệu dự án & Dataset",
            "⚡ Mô Hình Dự Đoán",
            "📊 Chỉ số thực nghiệm & Ma trận nhầm lẫn",
        ]
    )
    st.markdown("---")

# ==============================================================================
# TRANG 1: GIỚI THIỆU & THỐNG KÊ DATASET
# ==============================================================================
if page == "🏠 Giới thiệu dự án & Dataset":
    st.title("🔮 VietText Analyzer — NLP Research Dashboard")
    st.markdown("### Phân tích Sắc thái và Chủ đề Ý kiến Sinh viên bằng Machine Learning & Deep Learning")
    st.divider()

    st.header("1. Giới thiệu đề tài")
    st.write("""
    Hệ thống giải quyết đồng thời **hai bài toán NLP cốt lõi** trên khối văn bản ngôn ngữ tự nhiên:
    1. **Phân tích cảm xúc (Sentiment Analysis):** Nhận diện trạng thái tâm lý ý kiến sinh viên.
    2. **Phân loại chủ đề (Topic Classification):** Tự động bóc tách phân loại nhóm nội dung phản hồi.
    """) 
    
    st.markdown("""
    | Bài toán (Task) | Nhãn phân loại (Labels) | Tiếp cận công nghệ |
    |---|---|---|
    | **Phân tích sắc thái (Sentiment)** | Tiêu cực / Trung lập / Tích cực | TF-IDF + Linear ML, PhoBERT Transformer |
    | **Phân loại chủ đề (Topic)** | Curriculum / Facility / Lecturer / Other | TF-IDF + Linear ML |
    """)
    st.divider()

    st.header("2. Khám phá Bộ dữ liệu")
    @st.cache_data
    def load_dataset():
        _require_file(DATASET_EXCEL, "Dataset train.xlsx")
        df = pd.read_excel(DATASET_EXCEL)
        sent_map  = {0: "Tiêu cực 😡", 1: "Trung lập 😐", 2: "Tích cực 😍"}
        if len(df.columns) >= 1: df["sentence"]        = df.iloc[:, 0]
        if len(df.columns) >= 2: df["sentiment_label"] = df.iloc[:, 1].map(sent_map).fillna(df.iloc[:, 1])
        if len(df.columns) >= 3: df["topic_label"]     = df.iloc[:, 2].apply(topic_vi)
        df = df[df["sentence"].astype(str).str.lower() != "sentence"]
        return df

    df = load_dataset()
    st.subheader("📑 100 dòng dữ liệu đầu tiên (Xem trước cấu trúc)")
    display_cols = [c for c in ["sentence", "sentiment_label", "topic_label"] if c in df.columns]
    st.dataframe(df[display_cols].head(100), use_container_width=True)
    st.divider()

    st.header("3. Thống kê phân bố dữ liệu")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Phân bố Sắc thái (Sentiment Distribution)")
        if "sentiment_label" in df.columns:
            st.bar_chart(df["sentiment_label"].value_counts(), color="#ff4b4b")
    with col2:
        st.subheader("📊 Phân bố Chủ đề (Topic Distribution)")
        if "topic_label" in df.columns:
            st.bar_chart(df["topic_label"].value_counts(), color="#0068c9")

# ==============================================================================
# TRANG 2: Mô Hình Dự Đoán
# ==============================================================================
elif page == "⚡ Mô Hình Dự Đoán":
    st.title("⚡ Real-time Multi-Model Inference Dashboard")
    st.markdown("Nhập câu đánh giá học thuật — hệ thống phân tích sắc thái và chủ đề song song thời gian thực.")

    user_input = st.text_area(
        "✍️ Nhập nội dung ý kiến cần phân tích:",
        placeholder="Ví dụ: Thầy cô dạy rất nhiệt tình và dễ hiểu cơ mà phòng học hơi nóng...",
        height=120
    )

    if st.button(" phân tích ", type="primary"):
        if not user_input.strip():
            st.warning("⚠️ Vui lòng nhập nội dung trước khi bấm nút phân tích!")
        else:
            processed = preprocess_pipeline(user_input)
            with st.expander("🔍 Văn bản sau phân tách từ ghép"):
                st.code(processed, language="text")

            # ── Sentiment ──────────────────────────────────────────────────────
            st.subheader("📍 1. Kết quả phân tích Sắc thái (Sentiment)")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<div style='background-color:#f0f2f6;padding:10px;border-radius:5px;'><b>🔹 Phương pháp: TF-IDF + ML Baseline</b></div>", unsafe_allow_html=True)
                st.write("")
                try:
                    label, prob_dict = tfidf_predict_single_with_prob(processed, tfidf_sent_m, tfidf_sent_le)
                    label_to_display_with_details(label, prob_dict)
                except Exception as e:
                    st.error(f"Lỗi TF-IDF Sentiment: {e}")

            with col2:
                st.markdown("<div style='background-color:#e8f0fe;padding:10px;border-radius:5px;'><b>🔹 Phương pháp: PhoBERT Transformer (SOTA)</b></div>", unsafe_allow_html=True)
                st.write("")
                try:
                    label, prob_dict = phobert_predict_single_with_prob(processed, phobert_sent_m, phobert_sent_t, phobert_sent_le)
                    label_to_display_with_details(label, prob_dict)
                except Exception as e:
                    st.error(f"Lỗi PhoBERT Sentiment: {e}")

            # ── Topic ──────────────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("🎯 2. Kết quả phân loại Chủ đề (Topic)")
            has_phobert_topic = phobert_topic_m is not None
            col_t1, col_t2 = st.columns(2)

            with col_t1:
                st.markdown("<div style='background-color:#f0f2f6;padding:10px;border-radius:5px;'><b>🔹 Phương pháp: TF-IDF + ML Baseline</b></div>", unsafe_allow_html=True)
                st.write("")
                try:
                    label, prob_dict = tfidf_predict_single_with_prob(processed, tfidf_topic_m, tfidf_topic_le)
                    topic_to_display_with_details(label, prob_dict)
                except Exception as e:
                    st.error(f"Lỗi TF-IDF Topic: {e}")

            with col_t2:
                st.markdown("<div style='background-color:#e8f0fe;padding:10px;border-radius:5px;'><b>🔹 Phương pháp: PhoBERT Topic</b></div>", unsafe_allow_html=True)
                st.write("")
                if has_phobert_topic:
                    try:
                        label, prob_dict = phobert_predict_single_with_prob(processed, phobert_topic_m, phobert_topic_t, phobert_topic_le)
                        topic_to_display_with_details(label, prob_dict)
                    except Exception as e:
                        st.error(f"Lỗi PhoBERT Topic: {e}")
                else:
                    st.info("⏳ Mô hình PhoBERT riêng cho Topic chưa nạp — Sử dụng song song Baseline TF-IDF.")

# ==============================================================================
# TRANG 3: ĐÁNH GIÁ TOÀN BỘ TẬP VALIDATION
# ==============================================================================
elif page == "📊 Chỉ số thực nghiệm & Ma trận nhầm lẫn":
    st.title("📊 Model Performance — Live Evaluation")

    _require_file(VALID_PATH, "validation.xlsx")
    df_val     = pd.read_excel(VALID_PATH)
    df_val     = df_val[df_val.iloc[:, 0].astype(str).str.lower() != "sentence"].reset_index(drop=True)
    total_rows = len(df_val)
    st.info(f"📋 Tìm thấy tệp kiểm thử hợp lệ: **{total_rows}** mẫu dữ liệu thực tế.")

    sentences        = df_val.iloc[:, 0].values
    y_true_sent_raw  = df_val.iloc[:, 1].values
    y_true_topic_raw = df_val.iloc[:, 2].astype(str).str.strip().str.lower().values

    if st.button("Bắt đầu đánh giá toàn bộ tập validation 🚀", type="primary"):
        progress = st.progress(0)
        status   = st.empty()
        t0       = time.time()

        # Bước 1: Tiền xử lý
        status.text("⏳ Bước 1/4: Kích hoạt pipeline tiền xử lý...")
        cleaned = [preprocess_pipeline(str(s)) for s in sentences]
        progress.progress(10)

        # Bước 2: TF-IDF (cả 2 task cùng lúc)
        status.text("⏳ Bước 2/4: Mô hình toán học TF-IDF đang tính toán toán tử nền...")
        try:
            y_pred_tfidf_sent_raw  = tfidf_predict(cleaned, tfidf_sent_m,  tfidf_sent_le)
            y_pred_tfidf_topic_raw = tfidf_predict(cleaned, tfidf_topic_m, tfidf_topic_le)
        except Exception as e:
            st.error(f"❌ Lỗi xử lý khối TF-IDF: {e}")
            st.stop()
        progress.progress(35)

        # Bước 3: PhoBERT Sentiment
        PHOBERT_BATCH = 32
        total_batches = int(np.ceil(total_rows / PHOBERT_BATCH))
        y_pred_phobert_sent_raw = []
        for i in range(0, len(cleaned), PHOBERT_BATCH):
            b = i // PHOBERT_BATCH + 1
            status.text(f"⏳ Bước 3/4: PhoBERT Deep Learning — Xử lý Tensor Batch {b}/{total_batches}...")
            try:
                y_pred_phobert_sent_raw.extend(
                    phobert_predict_batch(cleaned[i:i+PHOBERT_BATCH], phobert_sent_m, phobert_sent_t, phobert_sent_le)
                )
            except Exception as e:
                st.error(f"❌ Lỗi PhoBERT Sentiment ở khối Tensor {b}: {e}")
                st.stop()
            progress.progress(35 + int((i / len(cleaned)) * 40))
        progress.progress(75)

        # Bước 4: PhoBERT Topic (nếu có)
        y_pred_phobert_topic_raw = []
        if phobert_topic_m is not None:
            for i in range(0, len(cleaned), PHOBERT_BATCH):
                b = i // PHOBERT_BATCH + 1
                status.text(f"⏳ Bước 4/4: PhoBERT Topic Deep Learning — Batch {b}/{total_batches}...")
                try:
                    y_pred_phobert_topic_raw.extend(
                        phobert_predict_batch(cleaned[i:i+PHOBERT_BATCH], phobert_topic_m, phobert_topic_t, phobert_topic_le)
                    )
                except Exception as e:
                    st.error(f"❌ Lỗi PhoBERT Topic ở khối Tensor {b}: {e}")
                    st.stop()
                progress.progress(75 + int((i / len(cleaned)) * 24))
        else:
            status.text("⏳ Bước 4/4: Không phát hiện file PhoBERT Topic — Tự động dùng chế độ bỏ qua...")

        progress.progress(100)
        status.success(f"🎉 Hệ thống tính toán hoàn tất trong thời gian kỷ lục: {time.time() - t0:.2f} giây!")

        # ── Chuẩn hoá nhãn sentiment ──────────────────────────────────────────
        y_true_sent         = standardize_sentiment(y_true_sent_raw)
        y_pred_tfidf_sent   = standardize_sentiment(y_pred_tfidf_sent_raw)
        y_pred_phobert_sent = standardize_sentiment(y_pred_phobert_sent_raw)

        # ── Chuẩn hoá nhãn topic ──────────────────────────────────────────────
        def std_topic(lst): return [str(l).strip().lower() for l in lst]
        y_true_topic         = std_topic(y_true_topic_raw)
        y_pred_tfidf_topic   = std_topic(y_pred_tfidf_topic_raw)
        y_pred_phobert_topic = std_topic(y_pred_phobert_topic_raw) if y_pred_phobert_topic_raw else []

        # ── Hàm vẽ confusion matrix ───────────────────────────────────────────
        def plot_cm(y_t, y_p, labels, tick_names, title, cmap="Blues"):
            cm  = confusion_matrix(y_t, y_p, labels=labels)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax, 
                        xticklabels=tick_names, yticklabels=tick_names,
                        cbar=True, annot_kws={"size": 11, "weight": "bold"})
            ax.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
            ax.set_ylabel("True Label", fontsize=10, labelpad=8)
            ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
            plt.xticks(rotation=35, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            return fig

        # ══════════════════════════════════════════════════════════════════════
        # PHẦN A: SENTIMENT
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.header("🔵 A. Phân tích Sắc thái (Sentiment Analysis)")

        acc_tf_s = accuracy_score(y_true_sent, y_pred_tfidf_sent) * 100
        f1_tf_s  = f1_score(y_true_sent, y_pred_tfidf_sent, average="weighted") * 100
        
        acc_ph_s = accuracy_score(y_true_sent, y_pred_phobert_sent) * 100
        f1_ph_s  = f1_score(y_true_sent, y_pred_phobert_sent, average="weighted") * 100

        st.subheader("📈 Chỉ số hiệu năng thực nghiệm")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("TF-IDF Accuracy", f"{acc_tf_s:.2f}%")
        m_col2.metric("TF-IDF F1-Score", f"{f1_tf_s:.2f}%")
        m_col3.metric("PhoBERT Accuracy 🔥", f"{acc_ph_s:.2f}%", f"+{(acc_ph_s - acc_tf_s):.2f}%")
        m_col4.metric("PhoBERT F1-Score 🔥", f"{f1_ph_s:.2f}%", f"+{(f1_ph_s - f1_tf_s):.2f}%")

        st.subheader("🧩 Biểu đồ ma trận toán học nhầm lẫn (Confusion Matrix)")
        SENT_LABELS = ["0", "1", "2"]
        SENT_TICKS  = ["Tiêu cực (Neg)", "Trung lập (Neu)", "Tích cực (Pos)"]
        sc1, sc2 = st.columns(2)
        with sc1:
            st.pyplot(plot_cm(y_true_sent, y_pred_tfidf_sent, SENT_LABELS, SENT_TICKS, "TF-IDF Sentiment Matrix", cmap="Blues"))
        with sc2:
            st.pyplot(plot_cm(y_true_sent, y_pred_phobert_sent, SENT_LABELS, SENT_TICKS, "PhoBERT Sentiment Matrix", cmap="Blues"))

        # ══════════════════════════════════════════════════════════════════════
        # PHẦN B: TOPIC
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.header("🟢 B. Phân loại Chủ đề (Topic Classification)")

        TOPIC_LABELS = sorted(set(y_true_topic))
        TOPIC_TICKS  = []
        for l in TOPIC_LABELS:
            ten_vi = topic_vi(l)
            if "chương trình" in ten_vi.lower():
                TOPIC_TICKS.append("Giáo trình")
            elif "cơ sở" in ten_vi.lower():
                TOPIC_TICKS.append("Cơ sở VC")
            elif "giảng viên" in ten_vi.lower() or "giảng dạy" in ten_vi.lower():
                TOPIC_TICKS.append("Giảng viên")
            else:
                TOPIC_TICKS.append("Chủ đề khác")

        acc_tf_t = accuracy_score(y_true_topic, y_pred_tfidf_topic) * 100
        f1_tf_t  = f1_score(y_true_topic, y_pred_tfidf_topic, average='weighted') * 100

        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        t_col1.metric("TF-IDF Topic Accuracy", f"{acc_tf_t:.2f}%")
        t_col2.metric("TF-IDF Topic F1-Score", f"{f1_tf_t:.2f}%")
        
        if y_pred_phobert_topic:
            acc_ph_t = accuracy_score(y_true_topic, y_pred_phobert_topic) * 100
            f1_ph_t  = f1_score(y_true_topic, y_pred_phobert_topic, average='weighted') * 100
            t_col3.metric("PhoBERT Topic Accuracy", f"{acc_ph_t:.2f}%")
            t_col4.metric("PhoBERT Topic F1-Score", f"{f1_ph_t:.2f}%")
        else:
            t_col3.metric("PhoBERT Topic Accuracy", "N/A")
            t_col4.metric("PhoBERT Topic F1-Score", "N/A")

        st.subheader("🧩 Biểu đồ ma trận toán học nhầm lẫn (Confusion Matrix)")
        tc1, tc2 = st.columns(2)
        with tc1:
            fig_tfidf = plot_cm(y_true_topic, y_pred_tfidf_topic, TOPIC_LABELS, TOPIC_TICKS, "TF-IDF Topic Matrix", cmap="Greens")
            st.pyplot(fig_tfidf)
        with tc2:
            if y_pred_phobert_topic:
                fig_phobert = plot_cm(y_true_topic, y_pred_phobert_topic, TOPIC_LABELS, TOPIC_TICKS, "PhoBERT Topic Matrix", cmap="Greens")
                st.pyplot(fig_phobert)
