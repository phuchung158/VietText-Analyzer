
import os
import torch
import streamlit as st
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="VietText Analyzer Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- ĐƯỜNG DẪN MÔ HÌNH LOCAL ---
MODEL_DIR = "./models/phobert"

# --- HÀM NẠP MÔ HÌNH (DÙNG CACHE) ---
@st.cache_resource
def load_phobert():
    if os.path.exists(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")):
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        status = "🎯 Đang chạy PhoBERT tối ưu từ Local (88.17%)"
    else:
        # Phương án chống cháy khi deploy trên Cloud chưa có file trọng số nặng
        DEFAULT_MODEL = "vinai/phobert-base"
        model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL, num_labels=3)
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
        status = "⚠️ Đang tải PhoBERT gốc từ Hugging Face (Chống cháy)"
    return model, tokenizer, status

# --- MENU ĐIỀU HƯỚNG TẠI SIDEBAR ---
st.sidebar.title("📌 Menu Dự Án")
page = st.sidebar.selectbox(
    "Di chuyển giữa các trang:",
    ["Trang 1: Giới thiệu & Dataset", "Trang 2: Dự đoán Cảm xúc (3 Mô hình)", "Trang 3: Phân tích Chủ đề (Topic)"]
)

st.sidebar.markdown("---")
st.sidebar.info("Đồ án: Phân tích ý kiến phản hồi của sinh viên bằng Học máy và Học sâu.")

# ==============================================================================
# TRANG 1: GIỚI THIỆU & DATASET
# ==============================================================================
if page == "Trang 1: Giới thiệu & Dataset":
    st.title("🔮 VietText Analyzer Project")
    st.subheader("Hệ thống phân loại văn bản tiếng Việt đa kiến trúc")
    
    st.markdown("""
    Chào mừng bạn đến với ứng dụng thử nghiệm **VietText Analyzer**. Dự án này được thực hiện nhằm so sánh hiệu năng giữa các kiến trúc xử lý ngôn ngữ tự nhiên (NLP) truyền thống và hiện đại trong bài toán phân tích ý kiến phản hồi của sinh viên.
    """)
    
    st.markdown("### 📊 Tổng quan về Bộ dữ liệu (Dataset)")
    st.write("Dự án sử dụng bộ dữ liệu **Synthetic Vietnamese Students Feedback Corpus** được chia sẵn làm 2 tập:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Tập Huấn luyện (Train Dataset)", value="`synthetic_train.csv`")
    with col2:
        st.metric(label="Tập Đánh giá (Validation Dataset)", value="`synthetic_val.csv`")
        
    st.markdown("""
    **Các trường thông tin cốt lõi cấu thành dữ liệu:**
    * `sentence`: Nội dung phản hồi bằng tiếng Việt của sinh viên (Ví dụ: *'Cơ sở vật chất nhà trường rất khang trang'*).
    * `sentiment`: Nhãn cảm xúc gồm 3 lớp: **Positive (Tích cực)**, **Neutral (Bình thường)**, **Negative (Tiêu cực)**.
    * `topic`: Chủ đề phản hồi (Ví dụ: *Cơ sở vật chất, Giảng viên, Chương trình học*).
    """)
    
    # Bảng so sánh hiệu năng tổng quan thu được từ Kaggle
    st.markdown("### 📈 Kết quả thực nghiệm trên tập Validation")
    data_perf = {
        "Mô hình (Architecture)": ["TF-IDF + Machine Learning", "LSTM + Word2Vec", "PhoBERT-base (VinAI)"],
        "Accuracy (Độ chính xác)": ["84.58%", "Đang cập nhật...", "88.17%"],
        "F1-Score": ["~84.50%", "Đang cập nhật...", "88.17%"],
        "Trạng thái": ["Baseline nhẹ, chạy nhanh", "Hiểu chuỗi mức trung bình", "Mạnh nhất, hiểu sâu ngữ cảnh"]
    }
    st.table(pd.DataFrame(data_perf))

# ==============================================================================
# TRANG 2: DỰ ĐOÁN CẢM XÚC (3 MÔ HÌNH)
# ==============================================================================
elif page == "Trang 2: Dự đoán Cảm xúc (3 Mô hình)":
    st.title("😡 😐 😍 Phân tích Cảm xúc Sinh viên")
    st.write("Nhập câu văn để chạy thử nghiệm đối sánh trực tiếp kết quả dự đoán giữa các mô hình.")
    
    # Chọn mô hình muốn dùng để dự đoán
    selected_model = st.radio(
        "Lựa chọn mô hình xử lý:",
        ["TF-IDF + Machine Learning", "LSTM + Word2Vec", "PhoBERT (Transformer)"],
        horizontal=True
    )
    
    user_input = st.text_area("Nhập câu phản hồi cần phân tích cảm xúc:", placeholder="Ví dụ: Giảng viên dạy rất hay và nhiệt tình nhưng phòng học hơi nóng...")
    
    if st.button("Chạy Dự Đoán Cảm Xúc 🚀", type="primary"):
        if user_input.strip() == "":
            st.warning("⚠️ Hãy nhập văn bản trước khi bấm nút dự đoán.")
        else:
            st.markdown(f"#### Kết quả từ mô hình: **{selected_model}**")
            
            # --- XỬ LÝ DỰ ĐOÁN THEO MÔ HÌNH ĐƯỢC CHỌN ---
            if selected_model == "PhoBERT (Transformer)":
                model, tokenizer, status = load_phobert()
                st.caption(status)
                
                # Tiến hành Tokenize và Predict
                inputs = tokenizer(user_input, padding=True, truncation=True, max_length=128, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                    prediction = torch.argmax(outputs.logits, dim=-1).item()
                
                # Ánh xạ kết quả nhãn
                id2label = {0: "Negative (Tiêu cực) 😡", 1: "Neutral (Bình thường) 😐", 2: "Positive (Tích cực) 😍"}
                
                if prediction == 2:
                    st.success(f"Kết quả: **{id2label[prediction]}**")
                    st.balloons()
                elif prediction == 1:
                    st.info(f"Kết quả: **{id2label[prediction]}**")
                else:
                    st.error(f"Kết quả: **{id2label[prediction]}**")
                    
            elif selected_model == "TF-IDF + Machine Learning":
                # Mockup/Giả lập logic hiển thị (Bạn nạp file .pkl của TF-IDF vào đây nếu cần)
                st.warning("🤖 Mô hình TF-IDF Baseline: Hệ thống phân tích dựa trên từ khóa đơn lẻ.")
                st.info("Kết quả mô phỏng: **Positive (Tích cực) 😍** (Vui lòng liên kết file pkl để chạy thực tế)")
                
            elif selected_model == "LSTM + Word2Vec":
                # Giả lập logic hiển thị cho LSTM
                st.warning("🧠 Mô hình LSTM Deep Learning: Hệ thống phân tích dựa trên chuỗi vector từ.")
                st.info("Kết quả mô phỏng: **Positive (Tích cực) 😍** (Vui lòng liên kết file h5/keras để chạy thực tế)")

# ==============================================================================
# TRANG 3: PHÂN TÍCH CHỦ ĐỀ (TOPIC)
# ==============================================================================
elif page == "Trang 3: Phân tích Chủ đề (Topic)":
    st.title("🎯 Phân tích Chủ đề Văn bản (Topic Classification)")
    st.write("Trang này phụ trách nhận diện xem ý kiến của sinh viên đang thuộc khía cạnh (Topic) nào của nhà trường.")
    
    user_input_topic = st.text_area("Nhập câu phản hồi cần trích xuất chủ đề:", placeholder="Ví dụ: Wifi của trường dạo này chập chờn quá, không đăng ký tín chỉ được...")
    
    if st.button("Phân Tích Chủ Đề 🔍", type="secondary"):
        if user_input_topic.strip() == "":
            st.warning("⚠️ Hãy nhập văn bản trước khi bấm nút phân tích.")
        else:
            st.markdown("#### Khía cạnh chủ đề được phát hiện:")
            
            # --- MOCKUP HÀM PHÂN TÍCH CHỦ ĐỀ ---
            # Bạn có thể huấn luyện 1 mô hình PhoBERT thứ 2 cho cột 'topic' hoặc dùng luật (Rule-based) tạm thời như dưới đây:
            text_lower = user_input_topic.lower()
            
            if any(w in text_lower for w in ["wifi", "phòng học", "bàn ghế", "thang máy", "cơ sở", "vật chất", "nóng"]):
                st.warning("🏫 Chủ đề: **Cơ sở vật chất & Trang thiết bị**")
            elif any(w in text_lower for w in ["thầy", "cô", "giảng viên", "thầy giáo", "cô giáo", "dạy", "nhiệt tình"]):
                st.success("👨‍🏫 Chủ đề: **Chất lượng Giảng viên & Giảng dạy**")
            elif any(w in text_lower for w in ["học phí", "tiền học", "kinh phí", "nộp tiền"]):
                st.error("💰 Chủ đề: **Tài chính & Học phí**")
            else:
                st.info("📝 Chủ đề: **Các hoạt động khác / Ý kiến chung**")
                
            st.caption("Lưu ý: Hệ thống đang phân tích dựa trên bộ từ khóa ngữ nghĩa tổng quan của trường học.")
