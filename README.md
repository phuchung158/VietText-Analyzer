# VietText Analyzer

Vietnamese text classification experiments with three different architectures: **TF-IDF + Machine Learning (Baseline)**, **LSTM + Word2Vec**, and **PhoBERT (State-of-the-art Transformer)**.

## Setup

```bash
python -m venv .venv
# Trên Windows:
.venv\Scripts\activate
# Trên macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt


Plaintext
dataset/
├── synthetic_train.csv
└── synthetic_val.csv


python baseline/train_tfidf.py


python lstm/train_lstm.py

models/phobert/

python phobert/train_phobert.py --target sentiment

import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Định nghĩa đường dẫn tới mô hình local
MODEL_DIR = "./models/phobert"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Mảng ánh xạ nhãn số ngược lại thành chữ
id2label = {0: "negative", 1: "neutral", 2: "positive"}

def predict_sentiment(texts):
    # Token hóa văn bản
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    
    # Dự đoán không tính gradient để tăng tốc độ
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        
    return [id2label[p.item()] for p in predictions]

# Chạy thử nghiệm
sentences = [
    "Trường học rất tốt và giảng viên nhiệt tình.",
    "Học phí quá đắt mà cơ sở vật chất lại tồi tàn."
]

labels = predict_sentiment(sentences)
for text, label in zip(sentences, labels):
    print(f"Text: {text} -> Sentiment: {label}")
# Output mong đợi: ['positive', 'negative']
