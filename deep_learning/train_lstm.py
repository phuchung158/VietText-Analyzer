import argparse
import json
import os
import pickle
import sys

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from deep_learning.lstm_w2v import LSTMModel, SequenceVectorizer, Word2VecBuilder
from utils.data_loader import DataLoader
from utils.evaluator import Evaluator
from utils.preprocessing import load_and_preprocess


def prepare_tokenized_texts(texts):
    return load_and_preprocess(texts)


def prepare_texts(texts):
    tokenized_texts = prepare_tokenized_texts(texts)
    return [" ".join(tokens) for tokens in tokenized_texts]


def save_label_encoder(label_encoder, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(label_encoder, file)


def load_label_encoder(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def save_metadata(payload, path):
    def _to_jsonable(value):
        if isinstance(value, dict):
            return {key: _to_jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_to_jsonable(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(_to_jsonable(payload), file, ensure_ascii=False, indent=2)


def validate_label_coverage(train_records, valid_records, target):
    train_labels = {str(record[target]).strip() for record in train_records}
    valid_labels = {str(record[target]).strip() for record in valid_records}
    unseen_labels = sorted(valid_labels - train_labels)

    if unseen_labels:
        raise ValueError(
            f"Validation split contains unseen {target} labels not present in train split: "
            f"{unseen_labels}"
        )


def evaluate_saved_model(model_dir, dataset_dir=None, target="sentiment"):
    if target not in {"sentiment", "topic"}:
        raise ValueError("target must be either 'sentiment' or 'topic'.")

    dataset_dir = dataset_dir or os.path.join(PROJECT_ROOT, "dataset")

    model_path = os.path.join(model_dir, f"lstm_{target}_model.keras")
    vectorizer_path = os.path.join(model_dir, f"lstm_{target}_vectorizer.pkl")
    encoder_path = os.path.join(model_dir, f"lstm_{target}_label_encoder.pkl")

    for required_path in (model_path, vectorizer_path, encoder_path):
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"Missing required artifact: {required_path}")

    print("=" * 60)
    print(f"Evaluating saved LSTM model in: {model_dir}")
    print("=" * 60)

    _, valid_records = DataLoader.load_project_splits(dataset_dir)
    valid_sentences = [record["sentence"] for record in valid_records]
    valid_texts = prepare_texts(valid_sentences)
    y_true = [record[target] for record in valid_records]

    vectorizer = SequenceVectorizer.load(vectorizer_path)
    label_encoder = load_label_encoder(encoder_path)

    X_valid = vectorizer.transform(valid_texts)

    model = LSTMModel(vocab_size=vectorizer.vocab_size, max_length=vectorizer.max_length)
    model.load(model_path)

    predictions = model.predict(X_valid)
    decoded_predictions = label_encoder.inverse_transform(predictions)

    metrics = Evaluator.evaluate(y_true, decoded_predictions, f"LSTM-{target} ({model_dir})")

    print(f"
Loaded model: {model_path}")
    print(f"Loaded vectorizer: {vectorizer_path}")
    print(f"Loaded label encoder: {encoder_path}")

    return {
        "model_dir": model_dir,
        "model_path": model_path,
        "vectorizer_path": vectorizer_path,
        "label_encoder_path": encoder_path,
        "metrics": metrics,
    }


def evaluate_multiple_saved_models(model_dirs, dataset_dir=None, target="sentiment"):
    results = []
    for model_dir in model_dirs:
        results.append(evaluate_saved_model(model_dir=model_dir, dataset_dir=dataset_dir, target=target))
    return results


def train_lstm(
    target="sentiment",
    dataset_dir=None,
    model_dir=None,
    max_words=20000,
    max_length=100,
    embedding_dim=128,
    lstm_units=128,
    dense_units=64,
    epochs=10,
    batch_size=32,
    learning_rate=1e-3,
    use_word2vec=False,
    trainable_embeddings=None,
    use_class_weight=True,
    w2v_vector_size=100,
    w2v_window=5,
    w2v_min_count=2,
):
    if target not in {"sentiment", "topic"}:
        raise ValueError("target must be either 'sentiment' or 'topic'.")

    dataset_dir = dataset_dir or os.path.join(PROJECT_ROOT, "dataset")
    model_dir = model_dir or os.path.join(PROJECT_ROOT, "models", "lstm")
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 60)
    print("LSTM Training")
    print("=" * 60)

    print("
[1/5] Loading dataset...")
    train_records, valid_records = DataLoader.load_project_splits(dataset_dir)
    print(f"Train samples: {len(train_records)}")
    print(f"Validation samples: {len(valid_records)}")

    print("
[2/5] Preparing labels...")
    validate_label_coverage(train_records, valid_records, target)
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform([record[target] for record in train_records])
    valid_labels = label_encoder.transform([record[target] for record in valid_records])
    print(f"Target: {target}")
    print(f"Classes: {list(label_encoder.classes_)}")

    class_weight = None
    if use_class_weight:
        classes = np.unique(train_labels)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
        class_weight = {int(class_id): float(weight) for class_id, weight in zip(classes, weights)}
        print(f"Class weights: {class_weight}")

    print("
[3/5] Preprocessing texts...")
    train_sentences = [record["sentence"] for record in train_records]
    valid_sentences = [record["sentence"] for record in valid_records]
    train_texts = prepare_texts(train_sentences)
    valid_texts = prepare_texts(valid_sentences)
    print("Preprocessing completed")

    print("
[4/5] Vectorizing texts...")
    vectorizer = SequenceVectorizer(max_words=max_words, max_length=max_length)
    X_train = vectorizer.fit_transform(train_texts)
    X_valid = vectorizer.transform(valid_texts)
    print(f"Vocabulary size: {vectorizer.vocab_size}")
    print(f"Max sequence length: {max_length}")

    embedding_matrix = None
    word2vec_path = None
    if use_word2vec:
        tokenized_train_texts = prepare_tokenized_texts(train_sentences)
        w2v_builder = Word2VecBuilder(
            vector_size=w2v_vector_size,
            window=w2v_window,
            min_count=w2v_min_count,
        )
        w2v_builder.train(tokenized_train_texts)
        embedding_matrix = w2v_builder.build_embedding_matrix(
            vectorizer.word_index,
            max_words=vectorizer.vocab_size,
        )
        word2vec_path = os.path.join(model_dir, f"lstm_{target}_word2vec.model")
        w2v_builder.save(word2vec_path)
        print("Word2Vec embedding matrix prepared")

    if trainable_embeddings is None:
        trainable_embeddings = not use_word2vec

    print("
[5/5] Training and evaluation...")
    model = LSTMModel(
        vocab_size=vectorizer.vocab_size,
        max_length=max_length,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        dense_units=dense_units,
        learning_rate=learning_rate,
        bidirectional=True,
    )
    model.build(
        num_classes=len(label_encoder.classes_),
        embedding_matrix=embedding_matrix,
        trainable_embeddings=trainable_embeddings,
    )
    history = model.train(
        X_train,
        train_labels,
        X_valid,
        valid_labels,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
    )

    predictions = model.predict(X_valid)
    decoded_valid_labels = label_encoder.inverse_transform(valid_labels)
    decoded_predictions = label_encoder.inverse_transform(predictions)
    metrics = Evaluator.evaluate(decoded_valid_labels, decoded_predictions, f"LSTM-{target}")

    model_path = os.path.join(model_dir, f"lstm_{target}_model.keras")
    vectorizer_path = os.path.join(model_dir, f"lstm_{target}_vectorizer.pkl")
    encoder_path = os.path.join(model_dir, f"lstm_{target}_label_encoder.pkl")
    metadata_path = os.path.join(model_dir, f"lstm_{target}_metadata.json")

    model.save(model_path)
    vectorizer.save(vectorizer_path)
    save_label_encoder(label_encoder, encoder_path)
    save_metadata(
        {
            "target": target,
            "classes": list(label_encoder.classes_),
            "max_words": max_words,
            "max_length": max_length,
            "embedding_dim": embedding_dim,
            "lstm_units": lstm_units,
            "dense_units": dense_units,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "use_word2vec": use_word2vec,
            "trainable_embeddings": trainable_embeddings,
            "use_class_weight": use_class_weight,
            "class_weight": class_weight,
            "word2vec_path": word2vec_path,
            "history": history.history,
            "metrics": metrics,
        },
        metadata_path,
    )

    print(f"
Saved model to: {model_path}")
    print(f"Saved vectorizer to: {vectorizer_path}")
    print(f"Saved label encoder to: {encoder_path}")
    print(f"Saved metadata to: {metadata_path}")
    if word2vec_path:
        print(f"Saved Word2Vec model to: {word2vec_path}")

    return {
        "model_path": model_path,
        "vectorizer_path": vectorizer_path,
        "label_encoder_path": encoder_path,
        "metadata_path": metadata_path,
        "word2vec_path": word2vec_path,
        "metrics": metrics,
        "classes": list(label_encoder.classes_),
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train or evaluate LSTM text classifier.")
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
        default=os.path.join(PROJECT_ROOT, "models", "lstm"),
        help="Directory to save trained artifacts.",
    )
    parser.add_argument(
        "--eval-model-dirs",
        nargs="+",
        help="One or more model directories to evaluate without retraining.",
    )
    parser.add_argument("--max-words", type=int, default=20000, help="Maximum vocabulary size.")
    parser.add_argument("--max-length", type=int, default=100, help="Maximum sequence length.")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--lstm-units", type=int, default=128, help="Number of LSTM units.")
    parser.add_argument("--dense-units", type=int, default=64, help="Hidden dense layer size.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for Adam.",
    )
    parser.add_argument(
        "--use-word2vec",
        action="store_true",
        help="Initialize embeddings from a train-only Word2Vec model.",
    )
    parser.add_argument(
        "--trainable-embeddings",
        action="store_true",
        help="Allow embedding weights to keep training after initialization.",
    )
    parser.add_argument(
        "--no-class-weight",
        action="store_true",
        help="Disable balanced class weighting during training.",
    )
    parser.add_argument(
        "--w2v-vector-size",
        type=int,
        default=100,
        help="Word2Vec vector size when --use-word2vec is enabled.",
    )
    parser.add_argument(
        "--w2v-window",
        type=int,
        default=5,
        help="Word2Vec context window when --use-word2vec is enabled.",
    )
    parser.add_argument(
        "--w2v-min-count",
        type=int,
        default=2,
        help="Minimum token frequency for Word2Vec when --use-word2vec is enabled.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    if args.eval_model_dirs:
        evaluate_multiple_saved_models(
            model_dirs=args.eval_model_dirs,
            dataset_dir=args.dataset_dir,
            target=args.target,
        )
    else:
        train_lstm(
            target=args.target,
            dataset_dir=args.dataset_dir,
            model_dir=args.model_dir,
            max_words=args.max_words,
            max_length=args.max_length,
            embedding_dim=args.embedding_dim,
            lstm_units=args.lstm_units,
            dense_units=args.dense_units,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_word2vec=args.use_word2vec,
            trainable_embeddings=args.trainable_embeddings if args.use_word2vec else None,
            use_class_weight=not args.no_class_weight,
            w2v_vector_size=args.w2v_vector_size,
            w2v_window=args.w2v_window,
            w2v_min_count=args.w2v_min_count,
        )
