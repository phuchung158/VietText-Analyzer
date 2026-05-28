# Tạo file tao_readme.py và chạy để sinh ra file README.md tự động

readme_content = """# VietText Analyzer

Vietnamese text classification experiments with three different architectures: **TF-IDF + Machine Learning (Baseline)**, **LSTM + Word2Vec**, and **PhoBERT (State-of-the-art Transformer)**.

## Setup

```bash
python -m venv .venv
# Trên Windows:
.venv\\Scripts\\activate
# Trên macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
