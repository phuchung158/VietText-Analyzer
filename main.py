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
# TẢI MÔ HÌNH VÀO CACHE RESOURSE
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

# ==============================================================================
# CHUẨN HOÁ NHÃN SENTIMENT
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

def label_to_display(label_text):
    t = str(label_text).upper()
    if any(w in t for w in ["POS", "TÍCH CỰC", "2", "POSITIVE"]):
        st.success("🎯 TÍCH CỰC 😍")
    elif any(w in t for w in ["NEG", "TIÊU CỰC", "0", "NEGATIVE"]):
        st.error("🎯 TIÊU CỰC 😡")
    else:
        st.warning("🎯 TRUNG LẬP 😐")

# ==============================================================================
# SideBar Điều hướng
# ==============================================================================
st.sidebar.title("🎮 Hệ Thống Điều Khiển")
page = st.sidebar.radio("Danh mục trang:", [
    "🏠 Giới thiệu dự án & Dataset",
    "⚡ Trình dự đoán song song",
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
    Hệ thống giải quyết đồng thời **hai bài toán NLP**:

    | Bài toán | Nhãn | Mô hình |
    |---|---|---|
    | Phân tích sắc thái (Sentiment) | Tiêu cực / Trung lập / Tích cực | TF-IDF + ML, PhoBERT |
    | Phân loại chủ đề (Topic) | Curriculum / Facility / Lecturer / Other | TF-IDF + ML       |

    *\*PhoBERT Topic đang được huấn luyện (nếu hệ thống phát hiện cấu trúc thư mục weights sẽ tự động kích hoạt).*
    """)

    st.header("2. Khám phá Bộ dữ liệu")

    @st.cache_data
    def load_dataset():
        _require_file(DATASET_EXCEL, "Dataset train.xlsx")
        df = pd.read_excel(DATASET_EXCEL)
        sent_map  = {0: "Tiêu cực", 1: "Trung lập", 2: "Tích cực"}
        if len(df.columns) >= 1: df["sentence"]        = df.iloc[:, 0]
        if len(df.columns) >= 2: df["sentiment_label"] = df.iloc[:, 1].map(sent_map).fillna(df.iloc[:, 1])
        if len(df.columns) >= 3: df["topic_label"]     = df.iloc[:, 2]
        df = df[df["sentence"].astype(str).str.lower() != "sentence"]
        return df

    df = load_dataset()
    st.subheader("📑 100 dòng dữ liệu đầu tiên")
    display_cols = [c for c in ["sentence", "sentiment_label", "topic_label"] if c in df.columns]
    st.dataframe(df[display_cols].head(100), use_container_width=True)
    st.divider()

    st.header("3. Thống kê phân bố dữ liệu")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Phân bố Sắc thái")
        if "sentiment_label" in df.columns:
            st.bar_chart(df["sentiment_label"].value_counts(), color="#ff4b4b")
    with col2:
        st.subheader("📊 Phân bố Chủ đề")
        if "topic_label" in df.columns:
            st.bar_chart(df["topic_label"].value_counts(), color="#0068c9")

# ==============================================================================
# TRANG 2: REAL-TIME INFERENCE DỰ ĐOÁN SONG SONG
# ==============================================================================
elif page == "⚡ Trình dự đoán song song":
    st.title("⚡ Real-time Multi-Model Inference Dashboard")
    st.markdown("Nhập câu đánh giá — hệ thống phân tích sắc thái và chủ đề song song.")

    user_input = st.text_area(
        "✍️ Nhập nội dung ý kiến cần phân tích:",
        placeholder="Ví dụ: Thầy cô dạy rất hay, cơ sở vật chất tốt...",
        height=100
    )

    if st.button("Kích hoạt phân tích 🚀", type="primary"):
        if not user_input.strip():
            st.warning("⚠️ Vui lòng nhập nội dung trước khi phân tích!")
        else:
            processed = " ".join(preprocess_pipeline(user_input))
            with st.expander("🔍 Văn bản sau tiền xử lý (PyVi)"):
                st.code(processed, language="text")

            # ── Sentiment ──────────────────────────────────────────────────────
            st.subheader("📍 1. Phân tích Sắc thái (Sentiment)")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🔹 TF-IDF + ML")
                try:
                    label = tfidf_predict([processed], tfidf_sent_m, tfidf_sent_le)[0]
                    label_to_display(label)
                except Exception as e:
                    st.error(f"Lỗi TF-IDF Sentiment: {e}")

            with col2:
                st.markdown("### 🔹 PhoBERT (SOTA)")
                try:
                    label = phobert_predict_batch([processed], phobert_sent_m, phobert_sent_t, phobert_sent_le)[0]
                    label_to_display(label)
                except Exception as e:
                    st.error(f"Lỗi PhoBERT Sentiment: {e}")

            # ── Topic ──────────────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("🎯 2. Phân loại Chủ đề (Topic)")
            has_phobert_topic = phobert_topic_m is not None
            col_t1, col_t2 = st.columns(2)

            with col_t1:
                st.markdown("### 🔹 TF-IDF + ML")
                try:
                    label = tfidf_predict([processed], tfidf_topic_m, tfidf_topic_le)[0]
                    st.info(f"📌 {topic_vi(label)}")
                except Exception as e:
                    st.error(f"Lỗi TF-IDF Topic: {e}")

            with col_t2:
                st.markdown("### 🔹 PhoBERT Topic")
                if has_phobert_topic:
                    try:
                        label = phobert_predict_batch([processed], phobert_topic_m, phobert_topic_t, phobert_topic_le)[0]
                        st.info(f"📌 {topic_vi(label)}")
                    except Exception as e:
                        st.error(f"Lỗi PhoBERT Topic: {e}")
                else:
                    st.info("⏳ Đang train trên Kaggle — sẽ tự động hiển thị khi hoàn tất cập nhật model.")

# ==============================================================================
# TRANG 3: ĐÁNH GIÁ TOÀN BỘ TẬP VALIDATION
# ==============================================================================
elif page == "📊 Chỉ số thực nghiệm & Ma trận nhầm lẫn":
    st.title("📊 Model Performance — Live Evaluation")

    _require_file(VALID_PATH, "validation.xlsx")
    df_val     = pd.read_excel(VALID_PATH)
    df_val     = df_val[df_val.iloc[:, 0].astype(str).str.lower() != "sentence"].reset_index(drop=True)
    total_rows = len(df_val)
    st.info(f"📋 Tập kiểm thử: **{total_rows}** mẫu — 2 bài toán: Sentiment & Topic")

    sentences        = df_val.iloc[:, 0].values
    y_true_sent_raw  = df_val.iloc[:, 1].values
    y_true_topic_raw = df_val.iloc[:, 2].astype(str).str.strip().str.lower().values

    if st.button("Bắt đầu đánh giá toàn bộ tập validation 🚀", type="primary"):
        progress = st.progress(0)
        status   = st.empty()
        t0       = time.time()

        # Bước 1: Tiền xử lý
        status.text("⏳ Bước 1/4: Tiền xử lý văn bản...")
        cleaned = [" ".join(preprocess_pipeline(str(s))) for s in sentences]
        progress.progress(10)

        # Bước 2: TF-IDF (cả 2 task cùng lúc)
        status.text("⏳ Bước 2/4: TF-IDF Sentiment & Topic đang suy luận...")
        try:
            y_pred_tfidf_sent_raw  = tfidf_predict(cleaned, tfidf_sent_m,  tfidf_sent_le)
            y_pred_tfidf_topic_raw = tfidf_predict(cleaned, tfidf_topic_m, tfidf_topic_le)
        except Exception as e:
            st.error(f"❌ Lỗi TF-IDF: {e}")
            st.stop()
        progress.progress(35)

        # Bước 3: PhoBERT Sentiment
        PHOBERT_BATCH = 32
        total_batches = int(np.ceil(total_rows / PHOBERT_BATCH))
        y_pred_phobert_sent_raw = []
        for i in range(0, len(cleaned), PHOBERT_BATCH):
            b = i // PHOBERT_BATCH + 1
            status.text(f"⏳ Bước 3/4: PhoBERT Sentiment — batch {b}/{total_batches}...")
            try:
                y_pred_phobert_sent_raw.extend(
                    phobert_predict_batch(cleaned[i:i+PHOBERT_BATCH], phobert_sent_m, phobert_sent_t, phobert_sent_le)
                )
            except Exception as e:
                st.error(f"❌ Lỗi PhoBERT Sentiment batch {b}: {e}")
                st.stop()
            progress.progress(35 + int((i / len(cleaned)) * 40))
        progress.progress(75)

        # Bước 4: PhoBERT Topic (nếu có)
        y_pred_phobert_topic_raw = []
        if phobert_topic_m is not None:
            for i in range(0, len(cleaned), PHOBERT_BATCH):
                b = i // PHOBERT_BATCH + 1
                status.text(f"⏳ Bước 4/4: PhoBERT Topic — batch {b}/{total_batches}...")
                try:
                    y_pred_phobert_topic_raw.extend(
                        phobert_predict_batch(cleaned[i:i+PHOBERT_BATCH], phobert_topic_m, phobert_topic_t, phobert_topic_le)
                    )
                except Exception as e:
                    st.error(f"❌ Lỗi PhoBERT Topic batch {b}: {e}")
                    st.stop()
                progress.progress(75 + int((i / len(cleaned)) * 24))
        else:
            status.text("⏳ Bước 4/4: PhoBERT Topic chưa sẵn sàng — bỏ qua...")

        progress.progress(100)
        status.success(f"🎉 Hoàn thành trong {time.time() - t0:.2f} giây!")

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
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax, xticklabels=tick_names, yticklabels=tick_names)
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("True", fontsize=8)
            ax.set_title(title, fontsize=9, fontweight="bold")
            plt.tight_layout()
            return fig

        # ══════════════════════════════════════════════════════════════════════
        # PHẦN A: SENTIMENT
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.header("🔵 A. Phân tích Sắc thái (Sentiment Analysis)")

        sent_results = []
        for name, yp in [
            ("TF-IDF + ML",             y_pred_tfidf_sent),
            ("PhoBERT Transformer (SOTA)", y_pred_phobert_sent),
        ]:
            acc = accuracy_score(y_true_sent, yp) * 100
            f1  = f1_score(y_true_sent, yp, average="weighted") * 100
            sent_results.append({
                "Mô hình": name,
                "Accuracy (%)": f"{acc:.2f}",
                "F1-Score Weighted (%)": f"{f1:.2f}",
            })
        st.subheader("📈 Chỉ số hiệu năng — Sentiment")
        st.table(pd.DataFrame(sent_results))

        st.subheader("🧩 Ma trận nhầm lẫn — Sentiment")
        SENT_LABELS = ["0", "1", "2"]
        SENT_TICKS  = ["Negative", "Neutral", "Positive"]
        sc1, sc2 = st.columns(2)
        with sc1:
            st.pyplot(plot_cm(y_true_sent, y_pred_tfidf_sent, SENT_LABELS, SENT_TICKS, "TF-IDF Sentiment", cmap="Blues"))
        with sc2:
            st.pyplot(plot_cm(y_true_sent, y_pred_phobert_sent, SENT_LABELS, SENT_TICKS, "PhoBERT Sentiment", cmap="Blues"))

        # ══════════════════════════════════════════════════════════════════════
        # PHẦN B: TOPIC
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.header("🟢 B. Phân loại Chủ đề (Topic Classification)")

        TOPIC_LABELS = sorted(set(y_true_topic))
        TOPIC_LABELS = sorted(set(y_true_topic))
        # 👇 ĐÃ SỬA: Lấy từ 2 từ đầu hoặc viết gọn lại để vừa khít biểu đồ 👇
        TOPIC_TICKS  = []
        for l in TOPIC_LABELS:
            ten_vi = topic_vi(l)
            if "chương trình" in ten_vi.lower():
                TOPIC_TICKS.append("Chương trình")
            elif "cơ sở" in ten_vi.lower():
                TOPIC_TICKS.append("Cơ sở VC")
            elif "giảng viên" in ten_vi.lower() or "giảng dạy" in ten_vi.lower():
                TOPIC_TICKS.append("Giảng viên")
            else:
                TOPIC_TICKS.append("Khác")

        topic_results = [{
            "Mô hình": "TF-IDF + ML (Topic)",
            "Accuracy (%)": f"{accuracy_score(y_true_topic, y_pred_tfidf_topic)*100:.2f}",
            "F1-Score Weighted (%)": f"{f1_score(y_true_topic, y_pred_tfidf_topic, average='weighted')*100:.2f}"
        }]

        if y_pred_phobert_topic:
            topic_results.append({
                "Mô hình": "PhoBERT Topic",
                "Accuracy (%)": f"{accuracy_score(y_true_topic, y_pred_phobert_topic)*100:.2f}",
                "F1-Score Weighted (%)": f"{f1_score(y_true_topic, y_pred_phobert_topic, average='weighted')*100:.2f}",
            })
        else:
            st.caption("⏳ PhoBERT Topic chưa sẵn sàng — chỉ hiển thị thông số đánh giá thực nghiệm của TF-IDF.")

        st.subheader("📈 Chỉ số hiệu năng — Topic")
        st.table(pd.DataFrame(topic_results))

        st.subheader("🧩 Ma trận nhầm lẫn — Topic")
        if y_pred_phobert_topic:
            tc1, tc2 = st.columns(2)
            with tc1:
                st.pyplot(plot_cm(y_true_topic, y_pred_tfidf_topic, TOPIC_LABELS, TOPIC_TICKS, "TF-IDF Topic", cmap="Greens"))
            with tc2:
                st.pyplot(plot_cm(y_true_topic, y_pred_phobert_topic, TOPIC_LABELS, TOPIC_TICKS, "PhoBERT Topic", cmap="Greens"))
        else:
            st.pyplot(plot_cm(y_true_topic, y_pred_tfidf_topic, TOPIC_LABELS, TOPIC_TICKS, "TF-IDF Topic", cmap="Greens"))
