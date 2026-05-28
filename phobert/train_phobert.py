import argparse
import json
import os
import pickle
import sys

from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phobert.phobert_classifier import PhoBERTClassifier
from utils.data_loader import DataLoader
from utils.evaluator import Evaluator
from utils.preprocessing import load_and_preprocess

DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "phobert")
BASE_MODEL_NAME = "vinai/phobert-base"


def prepare_texts(texts):
    tokenized_texts = load_and_preprocess(texts)
    return [" ".join(tokens) for tokens in tokenized_texts]


def save_label_encoder(label_encoder, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(label_encoder, file)


def save_metadata(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def validate_label_coverage(train_records, valid_records, target):
    train_labels = {str(record[target]).strip() for record in train_records}
    valid_labels = {str(record[target]).strip() for record in valid_records}
    unseen_labels = sorted(valid_labels - train_labels)

    if unseen_labels:
        raise ValueError(
            f"Validation split contains unseen {target} labels not present in train split: "
            f"{unseen_labels}"
        )


def train_phobert(
    target="sentiment",
    dataset_dir=None,
    model_dir=None,
    model_name=BASE_MODEL_NAME,
    max_length=128,
    epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    max_train_samples=None,
    max_valid_samples=None,
):
    if target not in {"sentiment", "topic"}:
        raise ValueError("target must be either 'sentiment' or 'topic'.")

    dataset_dir = dataset_dir or os.path.join(PROJECT_ROOT, "dataset")
    model_dir = model_dir or DEFAULT_MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 60)
    print("PhoBERT Training")
    print("=" * 60)

    print("\n[1/4] Loading dataset...")
    train_records, valid_records = DataLoader.load_project_splits(dataset_dir)
    if max_train_samples is not None:
        train_records = train_records[:max_train_samples]
    if max_valid_samples is not None:
        valid_records = valid_records[:max_valid_samples]
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
    classifier = PhoBERTClassifier(model_name=model_name, num_labels=len(label_encoder.classes_))
    history = classifier.train(
        train_texts,
        train_labels,
        valid_texts,
        valid_labels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
    )

    predictions = classifier.predict(valid_texts, batch_size=batch_size, max_length=max_length)
    decoded_valid_labels = label_encoder.inverse_transform(valid_labels)
    decoded_predictions = label_encoder.inverse_transform(predictions)
    metrics = Evaluator.evaluate(decoded_valid_labels, decoded_predictions, f"PhoBERT-{target}")

    model_path = model_dir
    encoder_path = os.path.join(model_dir, "label_encoder.pkl")
    metadata_path = os.path.join(model_dir, "metadata.json")

    classifier.save(model_path)
    save_label_encoder(label_encoder, encoder_path)
    save_metadata(
        {
            "target": target,
            "classes": list(label_encoder.classes_),
            "model_name": model_name,
            "max_length": max_length,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_train_samples": max_train_samples,
            "max_valid_samples": max_valid_samples,
            "history": history,
            "metrics": metrics,
        },
        metadata_path,
    )

    print(f"\nSaved model to: {model_path}")
    print(f"Saved label encoder to: {encoder_path}")
    print(f"Saved metadata to: {metadata_path}")

    return {
        "model_path": model_path,
        "label_encoder_path": encoder_path,
        "metadata_path": metadata_path,
        "metrics": metrics,
        "classes": list(label_encoder.classes_),
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train PhoBERT text classifier.")
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
        default=DEFAULT_MODEL_DIR,
        help="Directory to save/load PhoBERT artifacts.",
    )
    parser.add_argument(
        "--model-name",
        default=BASE_MODEL_NAME,
        help="Hugging Face model name or local path.",
    )
    parser.add_argument("--max-length", type=int, default=128, help="Maximum token length.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optionally limit train samples for quick experiments.",
    )
    parser.add_argument(
        "--max-valid-samples",
        type=int,
        default=None,
        help="Optionally limit validation samples for quick experiments.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train_phobert(
        target=args.target,
        dataset_dir=args.dataset_dir,
        model_dir=args.model_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_train_samples=args.max_train_samples,
        max_valid_samples=args.max_valid_samples,
    )
