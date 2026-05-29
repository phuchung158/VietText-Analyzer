import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- IMPORT PIPELINE TIỀN XỬ LÝ ---
from utils.preprocessing import preprocess_pipeline

# ==============================================================================
# CẤU HÌNH GIAO DIỆN
# ==============================================================================
st.set_page_config(page_title="VietText Analyzer Dashboard", page_icon="🚀", layout="wide")

# --- ĐƯỜNG DẪN MÔ HÌNH ---
MODEL_PHOBERT        = "./models/phobert"
LABEL_ENCODER_PHOBERT = "./models/phobert/label_encoder.pkl"

MODEL_TFIDF          = "./models/tfidf/sentiment/baseline_sentiment_model.pkl"
LABEL_ENCODER_TFIDF  = "./models/tfidf/sentiment/baseline_sentiment_label_encoder.pkl"


DATASET_EXCEL = "./dataset/train.xlsx"
VALID_PATH    = "./dataset/validation.xlsx"

# ==============================================================================
# ĐỊNH NGHĨA KIẾN TRÚC LSTM PYTORCH
# Lớp này phải khớp với kiến trúc lúc bạn huấn luyện.
# Nếu kiến trúc khác, hãy sửa class này cho đúng.
# ==============================================================================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 vì bidirectional

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(emb)
        # Ghép hidden state của 2 chiều cuối cùng
        out = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(out))


def _require_file(path: str, label: str):
    """Dừng app ngay nếu file bắt buộc không tồn tại."""
    if not os.path.exists(path):
        st.error(f"❌ Không tìm thấy file bắt buộc **{label}**:\n`{path}`\n\nVui lòng kiểm tra lại đường dẫn.")
        st.stop()


# ==============================================================================
# TẢI MÔ HÌNH — cache để không tải lại mỗi lần render
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
        st.error(f"❌ Lỗi khi load PhoBERT: {e}")
        st.stop()

    with open(LABEL_ENCODER_PHOBERT, "rb") as f:
        phobert_le = pickle.load(f)

    # ── 2. TF-IDF ──────────────────────────────────────────────────────────────
    _require_file(MODEL_TFIDF, "TF-IDF model")
    _require_file(LABEL_ENCODER_TFIDF, "TF-IDF label encoder")

    try:
        with open(MODEL_TFIDF, "rb") as f:
            tfidf_model = pickle.load(f)
        with open(LABEL_ENCODER_TFIDF, "rb") as f:
            tfidf_le = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Lỗi khi load TF-IDF: {e}")
        st.stop()

    # ── 3. LSTM PyTorch ────────────────────────────────────────────────────────
    _require_file(MODEL_LSTM_PATH,       "LSTM model (.pt)")
    _require_file(VOCAB_LSTM_PATH,       "LSTM vocab")
    _require_file(LABEL_ENCODER_LSTM_PATH, "LSTM label encoder")
    _require_file(LSTM_CONFIG_PATH,      "LSTM config")

    try:
        with open(VOCAB_LSTM_PATH, "rb") as f:
            lstm_vocab = pickle.load(f)          # dict: token -> idx
        with open(LABEL_ENCODER_LSTM_PATH, "rb") as f:
            lstm_le = pickle.load(f)
        with open(LSTM_CONFIG_PATH, "rb") as f:
            cfg = pickle.load(f)
            # cfg phải có: vocab_size, embed_dim, hidden_size, num_layers, num_classes, max_len

        lstm_model = LSTMClassifier(
            vocab_size  = cfg["vocab_size"],
            embed_dim   = cfg["embed_dim"],
            hidden_size = cfg["hidden_size"],
            num_layers  = cfg["num_layers"],
            num_classes = cfg["num_classes"],
        )
        state = torch.load(MODEL_LSTM_PATH, map_location=device)
        lstm_model.load_state_dict(state)
        lstm_model.to(device)
        lstm_model.eval()
        lstm_max_len = cfg["max_len"]
    except Exception as e:
        st.error(f"❌ Lỗi khi load LSTM PyTorch: {e}")
        st.stop()

    return (
        phobert_model, phobert_tokenizer, phobert_le,
        tfidf_model, tfidf_le,
        lstm_model, lstm_vocab, lstm_le, lstm_max_len,
        device
    )


(phobert_m, phobert_t, phobert_le,
 tfidf_m, tfidf_le,
 lstm_m, lstm_vocab, lstm_le, lstm_max_len,
 DEVICE) = load_all_models()


# ==============================================================================
# HELPER: LSTM — tokenize + pad + predict
# ==============================================================================
def lstm_texts_to_tensor(texts, vocab, max_len):
    """Chuyển list[str] đã tiền xử lý thành LongTensor (batch, max_len)."""
    unk_idx = vocab.get("<UNK>", 0)
    pad_idx = vocab.get("<PAD>", 0)
    result = []
    for text in texts:
        tokens = text.split()
        ids = [vocab.get(t, unk_idx) for t in tokens]
        if len(ids) >= max_len:
            ids = ids[:max_len]
        else:
            ids += [pad_idx] * (max_len - len(ids))
        result.append(ids)
    return torch.tensor(result, dtype=torch.long)


@torch.no_grad()
def lstm_predict_batch(texts):
    """Trả về list nhãn dự đoán (string) cho list văn bản đã tiền xử lý."""
    tensor = lstm_texts_to_tensor(texts, lstm_vocab, lstm_max_len).to(DEVICE)
    logits = lstm_m(tensor)
    pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
    return lstm_le.inverse_transform(pred_ids)


@torch.no_grad()
def phobert_predict_batch(texts, batch_size=32):
    """Trả về list nhãn dự đoán (string) cho PhoBERT, xử lý theo batch thật sự."""
    all_labels = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i: i + batch_size]
        inputs = phobert_t(
            chunk,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        logits = phobert_m(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
        labels = phobert_le.inverse_transform(pred_ids)
        all_labels.extend(labels)
    return all_labels


# ==============================================================================
# CHUẨN HOÁ NHÃN
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
# SIDEBAR
# ==============================================================================
st.sidebar.title("🎮 Hệ Thống Điều Khiển")
st.sidebar.markdown("Chọn tính năng hiển thị đồ án:")
page = st.sidebar.radio("Danh mục trang:", [
    "🏠 Giới thiệu dự án & Dataset",
    "⚡ Trình dự đoán song song tổng lực",
    "📊 Chỉ số thực nghiệm & Ma trận nhầm lẫn",
])

# ==============================================================================
# TRANG 1: GIỚI THIỆU & DATASET
# ==============================================================================
if page == "🏠 Giới thiệu dự án & Dataset":
    st.title("🔮 VietText Analyzer — NLP Research Dashboard")
    st.markdown("### Phân tích Sắc thái và Chủ đề Ý kiến Sinh viên bằng Machine Learning & Deep Learning")
    st.divider()

    st.header("1. Giới thiệu đề tài")
    st.write("""
    Hệ thống giải quyết đồng thời hai bài toán lõi trong xử lý ngôn ngữ tự nhiên sử dụng 3 phương pháp tiếp cận:
    - **TF-IDF + Machine Learning:** Mô hình cơ sở truyền thống nhanh, gọn nhẹ.
    - **LSTM + Word2Vec (PyTorch):** Mô hình mạng học sâu chuỗi thời gian nắm bắt cấu trúc câu.
    - **PhoBERT Transformer:** Mô hình ngôn ngữ lớn tiên tiến (SOTA) tối ưu cho tiếng Việt.
    """)

    st.header("2. Khám phá Bộ dữ liệu (Dataset Explorer)")

    @st.cache_data
    def load_dataset():
        _require_file(DATASET_EXCEL, "Dataset train.xlsx")
        df = pd.read_excel(DATASET_EXCEL)
        sent_map  = {0: "Tiêu cực", 1: "Trung lập", 2: "Tích cực"}
        topic_map = {0: "Chương trình đào tạo", 1: "Giảng viên", 2: "Cơ sở vật chất", 3: "Học phí & Khác"}
        if len(df.columns) >= 1: df["sentence"]       = df.iloc[:, 0]
        if len(df.columns) >= 2: df["sentiment_label"] = df.iloc[:, 1].map(sent_map).fillna(df.iloc[:, 1])
        if len(df.columns) >= 3: df["topic_label"]    = df.iloc[:, 2].map(topic_map).fillna(df.iloc[:, 2])
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
# TRANG 2: DỰ ĐOÁN SONG SONG
# ==============================================================================
elif page == "⚡ Trình dự đoán song song tổng lực":
    st.title("⚡ Real-time Multi-Model Inference Dashboard")
    st.markdown("Nhập câu đánh giá của sinh viên — hệ thống chạy song song trên cả 3 mô hình với dữ liệu **thực tế**.")

    user_input = st.text_area(
        "✍️ Nhập nội dung ý kiến cần phân tích:",
        placeholder="Ví dụ: Thầy cô dạy rất hay, cơ sở vật chất tốt...",
        height=100,
    )

    if st.button("Kích hoạt phân tích tổng lực 🚀", type="primary"):
        if not user_input.strip():
            st.warning("⚠️ Vui lòng nhập nội dung văn bản trước khi nhấn phân tích!")
        else:
            tokens = preprocess_pipeline(user_input)
            processed = " ".join(tokens)

            with st.expander("🔍 Văn bản sau tiền xử lý (PyVi)"):
                st.code(processed, language="text")

            st.subheader("📍 1. Kết quả dự đoán sắc thái (Sentiment)")
            col1, col2, col3 = st.columns(3)

            # ── TF-IDF ──────────────────────────────────────────────────────────
            with col1:
                st.markdown("### 🔹 TF-IDF + ML")
                try:
                    pred_code  = tfidf_m.predict([processed])[0]
                    pred_label = tfidf_le.inverse_transform([pred_code])[0] if tfidf_le else str(pred_code)
                    label_to_display(pred_label)
                except Exception as e:
                    st.error(f"Lỗi TF-IDF: {e}")

            # ── LSTM PyTorch ─────────────────────────────────────────────────────
            with col2:
                st.markdown("### 🔹 LSTM + Word2Vec")
                try:
                    pred_label = lstm_predict_batch([processed])[0]
                    label_to_display(pred_label)
                except Exception as e:
                    st.error(f"Lỗi LSTM: {e}")

            # ── PhoBERT ──────────────────────────────────────────────────────────
            with col3:
                st.markdown("### 🔹 PhoBERT (SOTA)")
                try:
                    pred_label = phobert_predict_batch([processed])[0]
                    label_to_display(pred_label)
                except Exception as e:
                    st.error(f"Lỗi PhoBERT: {e}")

            # ── Topic Analysis ────────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("🎯 2. Nhận diện Chủ đề Phản hồi (Topic Analysis)")
            topics_dict = {
                "Cơ sở vật chất & Thiết bị 🏫": [
                    "máy_lạnh", "điều_hòa", "phòng_học", "bàn_ghế", "wifi", "mạng",
                    "thang_máy", "nhà_vệ_sinh", "giữ_xe", "bãi_xe", "máy_chiếu",
                    "thiết_bị", "cơ_sở_vật_chất",
                ],
                "Chất lượng Giảng dạy & Giảng viên 👨‍🏫": [
                    "thầy", "cô", "giảng_viên", "giảng_dạy", "nhiệt_tình",
                    "kiến_thức", "giảng_bài", "dễ_hiểu", "khó_hiểu",
                    "môn_học", "học_tập", "truyền_đạt",
                ],
                "Học phí & Chính sách Tài chính 💰": [
                    "tiền_học", "học_phí", "đắt", "rẻ", "tăng_học_phí",
                    "nộp_tiền", "tài_chính", "kinh_phí", "học_bổng",
                ],
            }
            detected = [t for t, kws in topics_dict.items() if any(k in processed for k in kws)]
            if not detected:
                detected.append("Ý kiến chung / Chủ đề khác 📝")
            for t in detected:
                st.info(f"Chủ đề được nhận diện: **{t}**")


# ==============================================================================
# TRANG 3: ĐÁNH GIÁ TOÀN BỘ TẬP VALIDATION — DỮ LIỆU THẬT
# ==============================================================================
elif page == "📊 Chỉ số thực nghiệm & Ma trận nhầm lẫn":
    st.title("📊 Model Performance — Live Evaluation (Dữ liệu thật)")

    _require_file(VALID_PATH, "validation.xlsx")
    df_val = pd.read_excel(VALID_PATH)
    df_val = df_val[df_val.iloc[:, 0].astype(str).str.lower() != "sentence"]
    total_rows = len(df_val)
    st.info(f"📋 Tập kiểm thử: **{total_rows}** mẫu")

    sentences  = df_val.iloc[:, 0].values
    y_true_raw = df_val.iloc[:, 1].values

    if st.button("Bắt đầu đánh giá toàn bộ tập validation 🚀", type="primary"):
        progress = st.progress(0)
        status   = st.empty()
        t0       = time.time()

        # ── Bước 1: Tiền xử lý ────────────────────────────────────────────────
        status.text("⏳ Bước 1/4: Tiền xử lý văn bản...")
        cleaned = [" ".join(preprocess_pipeline(str(s))) for s in sentences]
        progress.progress(10)

        # ── Bước 2: TF-IDF ────────────────────────────────────────────────────
        status.text("⏳ Bước 2/4: TF-IDF đang suy luận...")
        try:
            y_pred_tfidf_raw = tfidf_m.predict(cleaned)
        except Exception as e:
            st.error(f"❌ Lỗi TF-IDF predict: {e}")
            st.stop()
        progress.progress(30)

        # ── Bước 3: LSTM PyTorch ───────────────────────────────────────────────
        status.text("⏳ Bước 3/4: LSTM đang suy luận...")
        try:
            LSTM_BATCH = 128
            y_pred_lstm_raw = []
            for i in range(0, len(cleaned), LSTM_BATCH):
                chunk = cleaned[i: i + LSTM_BATCH]
                y_pred_lstm_raw.extend(lstm_predict_batch(chunk))
                pct = 30 + int((i / len(cleaned)) * 20)
                progress.progress(min(pct, 50))
        except Exception as e:
            st.error(f"❌ Lỗi LSTM predict: {e}")
            st.stop()
        progress.progress(50)

        # ── Bước 4: PhoBERT batch thật ────────────────────────────────────────
        PHOBERT_BATCH = 32
        total_batches  = int(np.ceil(total_rows / PHOBERT_BATCH))
        y_pred_phobert_raw = []
        for i in range(0, len(cleaned), PHOBERT_BATCH):
            batch_idx = i // PHOBERT_BATCH + 1
            status.text(f"⏳ Bước 4/4: PhoBERT xử lý batch {batch_idx}/{total_batches}...")
            chunk = cleaned[i: i + PHOBERT_BATCH]
            try:
                y_pred_phobert_raw.extend(phobert_predict_batch(chunk, batch_size=PHOBERT_BATCH))
            except Exception as e:
                st.error(f"❌ Lỗi PhoBERT batch {batch_idx}: {e}")
                st.stop()
            pct = 50 + int((i / len(cleaned)) * 50)
            progress.progress(min(pct, 99))

        progress.progress(100)
        status.success(f"🎉 Hoàn thành trong {time.time() - t0:.2f} giây!")

        # ── Chuẩn hoá nhãn ────────────────────────────────────────────────────
        y_true           = standardize_labels(y_true_raw)
        y_pred_tfidf     = standardize_labels(y_pred_tfidf_raw)
        y_pred_lstm      = standardize_labels(y_pred_lstm_raw)
        y_pred_phobert   = standardize_labels(y_pred_phobert_raw)

        # ── Bảng kết quả ──────────────────────────────────────────────────────
        st.subheader("📈 Chỉ số hiệu năng thực tế")
        results = []
        for name, y_pred in [
            ("TF-IDF + ML", y_pred_tfidf),
            ("LSTM + Word2Vec (PyTorch)", y_pred_lstm),
            ("PhoBERT Transformer (SOTA)", y_pred_phobert),
        ]:
            acc = accuracy_score(y_true, y_pred) * 100
            f1  = f1_score(y_true, y_pred, average="weighted") * 100
            results.append({
                "Kiến trúc mô hình": name,
                "Accuracy (%)": f"{acc:.2f}",
                "F1-Score Weighted (%)": f"{f1:.2f}",
            })
        st.table(pd.DataFrame(results))

        # ── Confusion Matrix ───────────────────────────────────────────────────
        st.subheader("🧩 Ma trận nhầm lẫn (từ dự đoán thực tế)")
        LABELS     = ["0", "1", "2"]
        TICK_NAMES = ["Neg", "Neu", "Pos"]

        def plot_cm(y_t, y_p, title):
            cm  = confusion_matrix(y_t, y_p, labels=LABELS)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=TICK_NAMES, yticklabels=TICK_NAMES)
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("True", fontsize=8)
            ax.set_title(title, fontsize=9, fontweight="bold")
            plt.tight_layout()
            return fig

        c1, c2, c3 = st.columns(3)
        with c1:
            st.pyplot(plot_cm(y_true, y_pred_tfidf,   "TF-IDF Matrix"))
        with c2:
            st.pyplot(plot_cm(y_true, y_pred_lstm,    "LSTM Matrix"))
        with c3:
            st.pyplot(plot_cm(y_true, y_pred_phobert, "PhoBERT Matrix"))
