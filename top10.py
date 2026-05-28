import os
import pickle
import sys

from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = os.path.dirname(__file__)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baseline.tfidf_lr import BaselineModel
from baseline.train_baseline import prepare_texts, validate_label_coverage
from utils.data_loader import DataLoader
from utils.evaluator import Evaluator


def load_label_encoder(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def build_baseline_artifacts(target="sentiment", dataset_dir=None, max_features=5000):
    if target not in {"sentiment", "topic"}:
        raise ValueError("target must be either 'sentiment' or 'topic'.")

    dataset_dir = dataset_dir or os.path.join(PROJECT_ROOT, "dataset")

    train_records, valid_records = DataLoader.load_project_splits(dataset_dir)
    validate_label_coverage(train_records, valid_records, target)

    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform([record[target] for record in train_records])
    valid_labels = label_encoder.transform([record[target] for record in valid_records])

    train_texts = prepare_texts([record["sentence"] for record in train_records])
    valid_texts = prepare_texts([record["sentence"] for record in valid_records])

    model = BaselineModel(max_features=max_features)
    model.model.fit(train_texts, train_labels)

    predictions = model.model.predict(valid_texts)
    probabilities = model.model.predict_proba(valid_texts)

    decoded_valid_labels = label_encoder.inverse_transform(valid_labels)
    decoded_predictions = label_encoder.inverse_transform(predictions)
    metrics = Evaluator.evaluate(
        decoded_valid_labels,
        decoded_predictions,
        f"Baseline-{target}",
    )

    return {
        "model": model,
        "label_encoder": label_encoder,
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


def predict_texts(texts, target="sentiment", model_path=None, label_encoder_path=None, max_features=5000):
    if isinstance(texts, str):
        texts = [texts]

    processed_texts = prepare_texts(texts)

    if model_path and label_encoder_path:
        model = BaselineModel(max_features=max_features)
        model.load(model_path)
        label_encoder = load_label_encoder(label_encoder_path)
    else:
        result = build_baseline_artifacts(target=target, max_features=max_features)
        model = result["model"]
        label_encoder = result["label_encoder"]

    encoded_predictions = model.predict(processed_texts)
    probabilities = model.predict_proba(processed_texts)
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
    result = build_baseline_artifacts(target="sentiment")
    top_rows = get_top_correct_predictions(result, top_k=10)
    print_top_correct_predictions(top_rows)
