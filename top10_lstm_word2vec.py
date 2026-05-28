import os
import pickle
import sys

PROJECT_ROOT = os.path.dirname(__file__)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from deep_learning.lstm_w2v import LSTMModel, SequenceVectorizer
from utils.data_loader import DataLoader
from utils.evaluator import Evaluator
from utils.preprocessing import load_and_preprocess


def prepare_texts(texts):
    tokenized_texts = load_and_preprocess(texts)
    return [" ".join(tokens) for tokens in tokenized_texts]


def load_label_encoder(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def build_lstm_word2vec_artifacts(
    target="sentiment",
    dataset_dir=None,
    model_dir=None,
):
    if target not in {"sentiment", "topic"}:
        raise ValueError("target must be either 'sentiment' or 'topic'.")

    dataset_dir = dataset_dir or os.path.join(PROJECT_ROOT, "dataset")
    model_dir = model_dir or os.path.join(PROJECT_ROOT, "models", "lstm_word2vec")

    train_records, valid_records = DataLoader.load_project_splits(dataset_dir)

    label_encoder_path = os.path.join(model_dir, f"lstm_{target}_label_encoder.pkl")
    vectorizer_path = os.path.join(model_dir, f"lstm_{target}_vectorizer.pkl")
    model_path = os.path.join(model_dir, f"lstm_{target}_model.keras")

    label_encoder = load_label_encoder(label_encoder_path)
    vectorizer = SequenceVectorizer.load(vectorizer_path)
    model = LSTMModel(
        vocab_size=vectorizer.vocab_size,
        max_length=vectorizer.max_length,
    )
    model.load(model_path)

    valid_labels = label_encoder.transform([record[target] for record in valid_records])
    valid_texts = prepare_texts([record["sentence"] for record in valid_records])
    X_valid = vectorizer.transform(valid_texts)

    predictions = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)

    decoded_valid_labels = label_encoder.inverse_transform(valid_labels)
    decoded_predictions = label_encoder.inverse_transform(predictions)
    metrics = Evaluator.evaluate(
        decoded_valid_labels,
        decoded_predictions,
        f"LSTM-Word2Vec-{target}",
    )

    return {
        "model": model,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "train_records": train_records,
        "valid_records": valid_records,
        "valid_labels": valid_labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "metrics": metrics,
        "target": target,
    }


def get_top_correct_predictions(result, top_k=10):
    label_encoder = result["label_encoder"]
    valid_records = result["valid_records"]
    valid_labels = result["valid_labels"]
    predictions = result["predictions"]
    probabilities = result["probabilities"]

    rows = []
    for index, (record, true_label, pred_label, prob_vector) in enumerate(
        zip(valid_records, valid_labels, predictions, probabilities),
        start=1,
    ):
        if true_label != pred_label:
            continue

        confidence = float(prob_vector[pred_label])
        rows.append(
            {
                "index": index,
                "sentence": record["sentence"],
                "true_label": label_encoder.inverse_transform([true_label])[0],
                "predicted_label": label_encoder.inverse_transform([pred_label])[0],
                "confidence": confidence,
            }
        )

    rows.sort(key=lambda row: row["confidence"], reverse=True)
    return rows[:top_k]


def predict_texts(texts, target="sentiment", model_dir=None):
    if isinstance(texts, str):
        texts = [texts]

    model_dir = model_dir or os.path.join(PROJECT_ROOT, "models", "lstm_word2vec")
    label_encoder_path = os.path.join(model_dir, f"lstm_{target}_label_encoder.pkl")
    vectorizer_path = os.path.join(model_dir, f"lstm_{target}_vectorizer.pkl")
    model_path = os.path.join(model_dir, f"lstm_{target}_model.keras")

    label_encoder = load_label_encoder(label_encoder_path)
    vectorizer = SequenceVectorizer.load(vectorizer_path)
    model = LSTMModel(
        vocab_size=vectorizer.vocab_size,
        max_length=vectorizer.max_length,
    )
    model.load(model_path)

    processed_texts = prepare_texts(texts)
    X = vectorizer.transform(processed_texts)

    encoded_predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    decoded_predictions = label_encoder.inverse_transform(encoded_predictions)

    outputs = []
    for raw_text, predicted_label, prob_vector in zip(texts, decoded_predictions, probabilities):
        top_index = int(prob_vector.argmax())
        outputs.append(
            {
                "text": raw_text,
                "predicted_label": predicted_label,
                "confidence": float(prob_vector[top_index]),
                "probabilities": {
                    label: float(score)
                    for label, score in zip(label_encoder.classes_, prob_vector)
                },
            }
        )

    return outputs


def print_top_correct_predictions(rows):
    print("\nTop correct predictions")
    print("=" * 80)
    for row in rows:
        print(
            f"{row['index']:>4} | {row['confidence']:.6f} | "
            f"{row['true_label']} | {row['sentence']}"
        )


if __name__ == "__main__":
    result = build_lstm_word2vec_artifacts(target="sentiment")
    top_rows = get_top_correct_predictions(result, top_k=10)
    print_top_correct_predictions(top_rows)
