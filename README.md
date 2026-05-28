# Vietnamese Sentiment Analysis

Dự án xây dựng hệ thống phân tích cảm xúc tiếng Việt trên bộ dữ liệu phản hồi sinh viên tổng hợp (**Synthetic Vietnamese Students Feedback Corpus**) bằng 3 hướng tiếp cận:

* Machine Learning truyền thống (TF-IDF + Linear Models)
* Deep Learning (LSTM + Word2Vec)
* Transformer hiện đại (PhoBERT)

---

# 📂 Dataset

Bộ dữ liệu sử dụng cho cả 3 mô hình là **Synthetic Vietnamese Students Feedback Corpus**.

## Cấu trúc thư mục dữ liệu

```plaintext
dataset/
├── synthetic_train.csv
└── synthetic_val.csv
```

Mỗi file CSV bao gồm:

| Column      | Description                                      |
| ----------- | ------------------------------------------------ |
| `sentence`  | Văn bản phản hồi của sinh viên                   |
| `sentiment` | Nhãn cảm xúc (`negative`, `neutral`, `positive`) |

---

# 1️⃣ Baseline Model — TF-IDF + Machine Learning

Mô hình truyền thống sử dụng phương pháp thống kê tần suất từ để phân loại văn bản.

## ⚙️ Cơ chế hoạt động

* Văn bản được:

  * làm sạch dữ liệu
  * tách từ tiếng Việt
* Sau đó trích xuất đặc trưng bằng:

  * **TF-IDF (Term Frequency - Inverse Document Frequency)**
* Bộ phân loại:

  * `Logistic Regression`
  * hoặc `LinearSVC`

## 📈 Kết quả thực nghiệm

| Metric   | Score      |
| -------- | ---------- |
| Accuracy | **84.58%** |

### ✅ Ưu điểm

* Chạy rất nhanh
* Nhẹ
* Ít tốn tài nguyên

### ❌ Nhược điểm

* Không hiểu ngữ cảnh
* Không xử lý tốt từ đồng nghĩa
* Không nắm được ngữ nghĩa tiếng Việt

## 🚀 Train model

```bash
python baseline/train_tfidf.py
```

---

# 2️⃣ Deep Learning Model — LSTM + Word2Vec

Mô hình mạng học sâu kết hợp xử lý chuỗi và nhúng từ.

## ⚙️ Cơ chế hoạt động

* Sử dụng **Pre-trained Word2Vec tiếng Việt**
* Biến từ thành các vector embedding có ý nghĩa ngữ nghĩa
* Chuỗi vector được đưa qua:

  * `LSTM (Long Short-Term Memory)`

LSTM giúp mô hình học:

* quan hệ phụ thuộc xa
* ngữ cảnh câu
* cấu trúc chuỗi văn bản

## 📈 Đánh giá

Mô hình hiểu ngữ cảnh tốt hơn TF-IDF nhưng:

### ✅ Ưu điểm

* Hiểu chuỗi văn bản tốt hơn
* Học được ngữ nghĩa từ

### ❌ Nhược điểm

* Huấn luyện lâu hơn
* Cần nhiều dữ liệu hơn
* Cấu hình embedding phức tạp hơn

## 🚀 Train model

```bash
python lstm/train_lstm.py
```

---

# 3️⃣ Transformer Model — PhoBERT (State-of-the-art)

Mô hình mạnh nhất trong dự án, sử dụng kiến trúc Transformer.

PhoBERT là mô hình RoBERTa được pre-train chuyên biệt cho tiếng Việt bởi VinAI.

---

# 📦 Model Artifacts

Do kích thước trọng số rất lớn (~540MB), các file model không được commit lên GitHub.

Hãy giải nén:

```plaintext
phobert_model.zip
```

và đặt vào thư mục:

```plaintext
models/phobert/
```

## Các file bắt buộc

```plaintext
models/phobert/
├── model.safetensors
├── config.json
├── vocab.txt
├── bpe.codes
├── tokenizer_config.json
└── special_tokens_map.json
```

### Ý nghĩa

| File                      | Chức năng                             |
| ------------------------- | ------------------------------------- |
| `model.safetensors`       | Trọng số mô hình (~135 triệu tham số) |
| `config.json`             | Kiến trúc mạng và số lượng nhãn       |
| `vocab.txt`               | Từ điển tokenizer                     |
| `bpe.codes`               | Byte Pair Encoding                    |
| `tokenizer_config.json`   | Cấu hình tokenizer                    |
| `special_tokens_map.json` | Mapping special tokens                |

---

# 🚀 Training on Kaggle Multi-GPU

## Hyperparameters

```python
learning_rate = 2e-5
batch_size = 16
epochs = 5
```

## ⚠️ Tối ưu lưu trữ Kaggle

Kaggle giới hạn dung lượng ổ đĩa khoảng 20GB nên quá trình train sử dụng:

```python
save_strategy="no"
```

Sau khi train xong:

* model được nén trực tiếp bằng:

```python
shutil.make_archive(...)
```

để tránh lỗi:

```plaintext
OSError: [Errno 28] No space left on device
```

---

# 📈 Experimental Results

PhoBERT đạt hiệu năng tốt nhất tại **Epoch 5**.

| Metric   | Score      |
| -------- | ---------- |
| Accuracy | **88.17%** |
| F1-Score | **88.17%** |

## 📌 Nhận xét

* PhoBERT vượt trội rõ rệt so với TF-IDF
* Bắt đầu xuất hiện dấu hiệu overfitting nhẹ sau Epoch 3:

  * Validation Loss tăng từ `0.64` → `0.70`

---

# 🚀 Train PhoBERT

```bash
python phobert/train_phobert.py --target sentiment
```

---

# 📊 Performance Comparison

| Model Architecture        | Accuracy   | F1-Score   | Advantages                   | Disadvantages          |
| ------------------------- | ---------- | ---------- | ---------------------------- | ---------------------- |
| TF-IDF + Machine Learning | 84.58%     | ~84.50%    | Rất nhanh, nhẹ               | Không hiểu ngữ cảnh    |
| LSTM + Word2Vec           | Bổ sung    | Bổ sung    | Hiểu chuỗi văn bản tốt hơn   | Cần embedding phức tạp |
| PhoBERT-base (VinAI)      | **88.17%** | **88.17%** | Hiểu sâu ngữ cảnh tiếng Việt | Tốn tài nguyên GPU     |

---

# 🏆 Conclusion

* TF-IDF phù hợp cho các hệ thống nhẹ và cần tốc độ cao.
* LSTM cải thiện khả năng hiểu ngữ cảnh nhưng yêu cầu dữ liệu và thời gian huấn luyện lớn hơn.
* PhoBERT cho kết quả tốt nhất và là lựa chọn tối ưu cho bài toán phân tích cảm xúc tiếng Việt hiện đại.
