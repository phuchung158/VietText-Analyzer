# VietText Analyzer

Vietnamese text classification experiments with TF-IDF, LSTM, Word2Vec, and PhoBERT.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data

The training scripts expect:

- `dataset/train.xlsx`
- `dataset/validation.xlsx`

Each file should contain `sentence`, `sentiment`, and `topic` columns.

## PhoBERT Model

Large trained model artifacts are intentionally not committed to GitHub. Put the Kaggle-downloaded PhoBERT checkpoint in:

```text
models/phobert/
```

Expected files include `config.json`, `model.safetensors`, tokenizer files, `label_encoder.pkl`, and `metadata.json`.

## Train PhoBERT

```bash
python phobert/train_phobert.py --target sentiment
```

By default this saves artifacts to `models/phobert`.

## Use PhoBERT

```python
from phobert import PhoBERTClassifier

classifier = PhoBERTClassifier()
labels = classifier.predict_labels(["Truong hoc rat tot va giang vien nhiet tinh."])
print(labels)
```
