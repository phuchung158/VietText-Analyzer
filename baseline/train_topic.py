import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# 👇 IMPORT BỘ TIỀN XỬ LÝ ĐỂ ĐỒNG BỘ VỚI STREAMLIT 👇
from utils.preprocessing import preprocess_pipeline

# Đảm bảo đường dẫn thư mục lưu mô hình tồn tại chính xác
os.makedirs("./models/tfidf/topic", exist_ok=True)

# 1. Đọc dữ liệu huấn luyện thật từ file Excel
TRAIN_PATH = "./dataset/train.xlsx"
VALID_PATH = "./dataset/validation.xlsx"

print("⏳ Đang nạp dữ liệu huấn luyện...")
df_train = pd.read_excel(TRAIN_PATH)
df_valid = pd.read_excel(VALID_PATH)

# Lọc bỏ dòng tiêu đề thừa nếu có
df_train = df_train[df_train.iloc[:, 0].astype(str).str.lower() != 'sentence']
df_valid = df_valid[df_valid.iloc[:, 0].astype(str).str.lower() != 'sentence']

# Cột 1: Văn bản (X), Cột 3: Chủ đề (y)
X_train_raw = df_train.iloc[:, 0].astype(str).values
y_train_raw = df_train.iloc[:, 2].values

X_valid_raw = df_valid.iloc[:, 0].astype(str).values
y_valid_raw = df_valid.iloc[:, 2].values

# 👇 BƯỚC SỬA ĐỔI CỐT LÕI: TIỀN XỬ LÝ TOÀN BỘ DATA BẰNG PYVI TRƯỚC KHI TRAIN 👇
print("⏳ Đang tiền xử lý văn bản bằng Pipeline PyVi (Đồng bộ hệ thống)...")
X_train = [" ".join(preprocess_pipeline(text)) for text in X_train_raw]
X_valid = [" ".join(preprocess_pipeline(text)) for text in X_valid_raw]

# 2. Chuẩn hóa nhãn chủ đề
print("⏳ Đang xử lý mã hóa nhãn chủ đề...")
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw.astype(str))
y_valid = le.transform(y_valid_raw.astype(str))

# 3. Xây dựng Pipeline: Tăng max_features để bao phủ hết từ ghép PyVi
print("🚀 Bắt đầu quá trình huấn luyện TF-IDF cho tác vụ phân loại Chủ đề...")
topic_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=10000)), # Tăng lên 10k features cho n-gram từ ghép
    ('clf', LogisticRegression(C=2.0, max_iter=1000, class_weight='balanced'))
])

# Huấn luyện mô hình
topic_pipeline.fit(X_train, y_train)

# 4. Đánh giá kiểm thử độ chính xác thực tế trên tập Validation
y_pred = topic_pipeline.predict(X_valid)
acc = accuracy_score(y_valid, y_pred)

print("\n" + "="*50)
print(f"🎉 HUẤN LUYỆN THÀNH CÔNG!")
print(f"📈 Độ chính xác thực tế sau đồng bộ (Accuracy): {acc * 100:.2f}%")
print("="*50)
print(classification_report(y_valid, y_pred, target_names=le.classes_))

# 5. Lưu mô hình mô hình và bộ mã hóa nhãn vào thư mục models/tfidf/topic/
MODEL_TOPIC_PATH = "./models/tfidf/topic/baseline_topic_model.pkl"
LABEL_TOPIC_PATH = "./models/tfidf/topic/baseline_topic_label_encoder.pkl"

with open(MODEL_TOPIC_PATH, 'wb') as f:
    pickle.dump(topic_pipeline, f)

with open(LABEL_TOPIC_PATH, 'wb') as f:
    pickle.dump(le, f)

print(f"💾 Đã đóng gói và lưu mô hình chủ đề tại: {MODEL_TOPIC_PATH}")