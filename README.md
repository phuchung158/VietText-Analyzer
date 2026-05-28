Markdown# VietText Analyzer

Vietnamese text classification experiments with three different architectures: **TF-IDF + Machine Learning (Baseline)**, **LSTM + Word2Vec**, and **PhoBERT (State-of-the-art Transformer)**.

## Setup

```bash
python -m venv .venv
# Trên Windows:
.venv\Scripts\activate
# Trên macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
DataBộ dữ liệu sử dụng cho cả 3 mô hình là Synthetic Vietnamese Students Feedback Corpus. Cấu trúc thư mục dữ liệu đầu vào mong đợi (đã chia sẵn tập Train/Val):Plaintextdataset/
├── synthetic_train.csv
└── synthetic_val.csv
Mỗi file chứa các cột dữ liệu gốc: sentence (văn bản phản hồi) và sentiment (nhãn cảm xúc: negative, neutral, positive).1. Mô hình Baseline: TF-IDF + Machine LearningMô hình truyền thống sử dụng phương pháp thống kê tần suất từ để phân loại văn bản.Cơ chế: Văn bản được làm sạch, tách từ tiếng Việt, sau đó trích xuất đặc trưng bằng kỹ thuật TF-IDF (Term Frequency-Inverse Document Frequency). Các thuật toán phân loại như Logistic Regression hoặc LinearSVC được áp dụng làm Baseline.Kết quả thực nghiệm: Đạt độ chính xác Accuracy: 84.58%. Mô hình này chạy rất nhanh, nhẹ nhưng điểm yếu là không hiểu được ngữ cảnh hoặc từ đồng nghĩa trong tiếng Việt.Kịch bản chạy:Bashpython baseline/train_tfidf.py
2. Mô hình Deep Learning: LSTM + Word2VecMô hình mạng học sâu kết hợp xử lý chuỗi và nhúng từ (Word Embedding).Cơ chế: Sử dụng một bộ nhúng từ tiền huấn luyện (Pre-trained Word2Vec cho tiếng Việt) để biến các từ thành các vector không gian có nghĩa. Sau đó, chuỗi vector này được đưa qua các tầng mạng LSTM (Long Short-Term Memory) để học các đặc trưng phụ thuộc xa (long-term dependencies) trong câu văn.Kết quả thực nghiệm: Cải thiện khả năng hiểu ngữ cảnh tốt hơn so với TF-IDF nhưng đòi hỏi thời gian huấn luyện lâu hơn và cần lượng dữ liệu lớn để tối ưu hóa trọng số.Kịch bản chạy:Bashpython lstm/train_lstm.py
3. Mô hình Transformer: PhoBERT (State-of-the-art)Mô hình mạnh nhất dựa trên kiến trúc RoBERTa được pre-train chuyên biệt cho tiếng Việt bởi VinAI.📦 Lưu ý về Trọng số Mô hình (Model Artifacts)Do kích thước trọng số của Transformer rất lớn (~540MB) nên các file này không được commit lên GitHub. Hãy giải nén file phobert_model.zip tải về từ Kaggle và đặt vào đúng thư mục sau:Plaintextmodels/phobert/
Các file bắt buộc phải có bao gồm:model.safetensors (Trọng số cốt lõi lưu trữ 135 triệu tham số của PhoBERT)config.json (Cấu hình kiến trúc mạng gồm 12 tầng mạng và hệ thống 3 nhãn đầu ra)Các file cấu hình bộ mã hóa ngôn ngữ: vocab.txt, bpe.codes, tokenizer_config.json, special_tokens_map.json.🚀 Quá trình huấn luyện trên Kaggle Multi-GPUTham số tối ưu: learning_rate=2e-5, batch_size=16, epochs=5.Cơ chế đặc biệt: Bật save_strategy="no" để chống tràn ổ đĩa 20GB của Kaggle, sau đó nén trực tiếp bằng lệnh Python (shutil.make_archive) thành file đơn lẻ để tải về máy local nhằm tránh lỗi OSError: [Errno 28] No space left on device.Kết quả thực nghiệm: Đạt đỉnh hiệu năng tối ưu vượt trội ở Epoch 5 với Accuracy đạt 88.17% và F1-Score đạt 88.17% (Xuất sắc đè bẹp mốc 84.58% của TF-IDF). Tín hiệu Overfitting nhẹ bắt đầu xuất hiện sau Epoch 3 khi Validation Loss tăng từ 0.64 lên 0.70.Kịch bản chạy huấn luyện tự động:Bashpython phobert/train_phobert.py --target sentiment
📊 Bảng So Sánh Hiệu Năng (Tập Validation)Mô hình (Model Architecture)AccuracyF1-ScoreƯu điểmNhược điểmTF-IDF + Machine Learning84.58%~84.50%Chạy cực nhanh, tốn rất ít tài nguyên phần cứng.Không hiểu ngữ cảnh, cú pháp và ngữ nghĩa từ.LSTM + Word2VecBổ sung % nếu cóBổ sung %Hiểu được chuỗi văn bản và ngữ nghĩa các từ đơn lẻ.Cần cấu hình mảng nhúng từ khá phức tạp.PhoBERT-base (VinAI)88.17%88.17%Hiểu sâu ngữ cảnh tiếng Việt, độ chính xác cao nhất.
