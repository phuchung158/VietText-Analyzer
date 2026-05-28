import argparse
import os
import pickle
import sys

from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baseline.tfidf_lr import BaselineModel
from utils.data_loader import DataLoader
from utils.evaluator import Evaluator
from utils.preprocessing import load_and_preprocess


def prepare_texts(texts):
    tokenized_texts = load_and_preprocess(texts)
    return [" ".join(tokens) for tokens in tokenized_texts]


def save_label_encoder(label_encoder, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(label_encoder, file)


def validate_label_coverage(train_records, valid_records, target):
    train_labels = {str(record[target]).strip() for record in train_records}
    valid_labels = {str(record[target]).strip() for record in valid_records}
    unseen_labels = sorted(valid_labels - train_labels)

    if unseen_labels:
        raise ValueError(
            f"Validation split contains unseen {target} labels not present in train split: "
            f"{unseen_labels}"
        )


def train_baseline(target="sentiment", dataset_dir=None, model_dir=None, max_features=5000):
    if target not in {"sentiment", "topic"}:
        raise ValueError("target must be either 'sentiment' or 'topic'.")

    dataset_dir = dataset_dir or os.path.join(PROJECT_ROOT, "dataset")
    model_dir = model_dir or os.path.join(PROJECT_ROOT, "models", "tfidf")
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 60)
    print("Baseline Training (TF-IDF + Logistic Regression)")
    print("=" * 60)

    print("\n[1/4] Loading dataset...")
    train_records, valid_records = DataLoader.load_project_splits(dataset_dir)
    print(f"Train samples: {len(train_records)}")
    print(f"Validation samples: {len(valid_records)}")

    print("\n[2/4] Preparing labels...")
    validate_label_coverage(train_records, valid_records, target)
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform([record[target] for record in train_records])
    valid_labels = label_encoder.transform([record[target] for record in valid_records])
    print(f"Target: {target}")
    print(f"Classes: {list(label_encoder.classes_)}")

    print("\n[3/4] Preprocessing texts...")
    train_texts = prepare_texts([record["sentence"] for record in train_records])
    valid_texts = prepare_texts([record["sentence"] for record in valid_records])
    print("Preprocessing completed")

    print("\n[4/4] Training and evaluation...")
    model = BaselineModel(max_features=max_features)
    model.train(train_texts, train_labels)
    predictions = model.predict(valid_texts)
    decoded_valid_labels = label_encoder.inverse_transform(valid_labels)
    decoded_predictions = label_encoder.inverse_transform(predictions)
    metrics = Evaluator.evaluate(
        decoded_valid_labels,
        decoded_predictions,
        f"Baseline-{target}",
    )

    model_path = os.path.join(model_dir, f"baseline_{target}_model.pkl")
    encoder_path = os.path.join(model_dir, f"baseline_{target}_label_encoder.pkl")
    model.save(model_path)
    save_label_encoder(label_encoder, encoder_path)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved label encoder to: {encoder_path}")

    return {
        "model_path": model_path,
        "label_encoder_path": encoder_path,
        "metrics": metrics,
        "classes": list(label_encoder.classes_),
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train baseline TF-IDF + Logistic Regression model.")
    parser.add_argument(
        "--target",
        choices=["sentiment", "topic"],
        default="sentiment",
        help="Label column to train on.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=os.path.join(PROJECT_ROOT, "dataset"),
        help="Directory containing train.xlsx and validation.xlsx.",
    )
    parser.add_argument(
        "--model-dir",
        default=os.path.join(PROJECT_ROOT, "models", "tfidf"),
        help="Directory to save trained artifacts.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum number of TF-IDF features.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train_baseline(
        target=args.target,
        dataset_dir=args.dataset_dir,
        model_dir=args.model_dir,
        max_features=args.max_features,
    )
