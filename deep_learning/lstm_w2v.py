import os
import pickle

import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPool1D,
    SpatialDropout1D,
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class Word2VecBuilder:
    def __init__(self, vector_size=100, window=5, min_count=2, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
        self.vocab_size = 0

    def train(self, tokenized_texts):
        """Train a Word2Vec model on tokenized texts."""
        print("Training Word2Vec...")
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
        )
        self.vocab_size = len(self.model.wv)
        print(f"Word2Vec trained. Vocab size: {self.vocab_size}")

    def build_embedding_matrix(self, word_index, max_words=None):
        """Create an embedding matrix aligned to a tokenizer word index."""
        if self.model is None:
            raise ValueError("Word2Vec model has not been trained or loaded.")

        max_index = max(word_index.values(), default=0)
        vocab_size = min(max_words or max_index + 1, max_index + 1)
        embedding_matrix = np.zeros((vocab_size, self.vector_size), dtype=np.float32)

        for word, index in word_index.items():
            if index >= vocab_size or word not in self.model.wv:
                continue
            embedding_matrix[index] = self.model.wv[word]

        return embedding_matrix

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load(self, path):
        self.model = Word2Vec.load(path)
        self.vocab_size = len(self.model.wv)


class SequenceVectorizer:
    def __init__(self, max_words=20000, max_length=100, oov_token="<OOV>"):
        self.max_words = max_words
        self.max_length = max_length
        self.oov_token = oov_token
        self.tokenizer = Tokenizer(
            num_words=max_words,
            filters="",
            lower=False,
            split=" ",
            oov_token=oov_token,
        )

    @property
    def vocab_size(self):
        learned_vocab_size = len(self.tokenizer.word_index) + 1
        if self.max_words is None:
            return learned_vocab_size
        return min(self.max_words, learned_vocab_size)

    @property
    def word_index(self):
        return self.tokenizer.word_index

    def fit(self, texts):
        self.tokenizer.fit_on_texts(texts)

    def transform(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding="post",
            truncating="post",
        )

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(
                {
                    "max_words": self.max_words,
                    "max_length": self.max_length,
                    "oov_token": self.oov_token,
                    "tokenizer": self.tokenizer,
                },
                file,
            )

    @classmethod
    def load(cls, path):
        with open(path, "rb") as file:
            payload = pickle.load(file)

        vectorizer = cls(
            max_words=payload["max_words"],
            max_length=payload["max_length"],
            oov_token=payload["oov_token"],
        )
        vectorizer.tokenizer = payload["tokenizer"]
        return vectorizer


class LSTMModel:
    def __init__(
        self,
        vocab_size,
        max_length=100,
        embedding_dim=128,
        lstm_units=128,
        dense_units=64,
        learning_rate=1e-3,
        bidirectional=True,
        spatial_dropout_rate=0.2,
        dense_dropout_rate=0.3,
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        self.spatial_dropout_rate = spatial_dropout_rate
        self.dense_dropout_rate = dense_dropout_rate
        self.model = None

    def build(self, num_classes, embedding_matrix=None, trainable_embeddings=True):
        """Build the LSTM classifier."""
        embedding_output_dim = (
            embedding_matrix.shape[1] if embedding_matrix is not None else self.embedding_dim
        )
        embedding_kwargs = {
            "input_dim": self.vocab_size,
            "output_dim": embedding_output_dim,
            "input_length": self.max_length,
            "mask_zero": True,
            "trainable": trainable_embeddings,
        }
        if embedding_matrix is not None:
            embedding_kwargs["weights"] = [embedding_matrix]

        recurrent_layer = LSTM(
            self.lstm_units,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.0,
        )
        if self.bidirectional:
            recurrent_layer = Bidirectional(recurrent_layer)

        self.model = Sequential(
            [
                Embedding(**embedding_kwargs),
                SpatialDropout1D(self.spatial_dropout_rate),
                recurrent_layer,
                GlobalMaxPool1D(),
                Dense(self.dense_units, activation="relu"),
                Dropout(self.dense_dropout_rate),
                Dense(num_classes, activation="softmax"),
            ]
        )
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def train(
        self,
        X_train,
        y_train,
        X_valid,
        y_valid,
        epochs=10,
        batch_size=32,
        patience=3,
        lr_patience=1,
        class_weight=None,
    ):
        """Train the LSTM classifier."""
        if self.model is None:
            raise ValueError("Model has not been built.")

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=lr_patience,
                min_lr=1e-5,
                verbose=1,
            ),
        ]

        print("Training LSTM model...")
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_valid, y_valid),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )
        print("LSTM training completed")
        return history

    def predict(self, X):
        """Predict encoded labels."""
        if self.model is None:
            raise ValueError("Model has not been built or loaded.")
        probabilities = self.model.predict(X, verbose=0)
        return np.argmax(probabilities, axis=1)

    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model has not been built or loaded.")
        return self.model.predict(X, verbose=0)

    def save(self, path):
        """Save the trained Keras model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load(self, path):
        """Load a saved Keras model."""
        self.model = load_model(path)
