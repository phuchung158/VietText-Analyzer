import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "phobert")
BASE_MODEL_NAME = "vinai/phobert-base"


class PhoBERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class PhoBERTClassifier:
    def __init__(self, model_name=None, num_labels=None, label_encoder_path=None):
        self.model_name = model_name or self._default_model_name()
        self.num_labels = num_labels
        self.label_encoder = self._load_label_encoder(label_encoder_path, self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if num_labels is not None:
            model_kwargs["num_labels"] = num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, **model_kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @staticmethod
    def _default_model_name():
        if os.path.exists(os.path.join(DEFAULT_MODEL_DIR, "config.json")):
            return DEFAULT_MODEL_DIR
        return BASE_MODEL_NAME

    @staticmethod
    def _load_label_encoder(label_encoder_path, model_name):
        candidates = []
        if label_encoder_path:
            candidates.append(label_encoder_path)
        if os.path.isdir(model_name):
            candidates.extend(
                [
                    os.path.join(model_name, "label_encoder.pkl"),
                    os.path.join(model_name, "phobert_sentiment_label_encoder.pkl"),
                ]
            )

        for path in candidates:
            if os.path.exists(path):
                with open(path, "rb") as file:
                    return pickle.load(file)
        return None

    def prepare_data(self, texts, labels=None, batch_size=16, max_length=256, shuffle=False):
        dataset = PhoBERTDataset(texts, labels, self.tokenizer, max_length=max_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train(
        self,
        train_texts,
        train_labels,
        valid_texts,
        valid_labels,
        epochs=3,
        batch_size=16,
        learning_rate=2e-5,
        max_length=256,
        warmup_ratio=0.1,
    ):
        train_loader = self.prepare_data(
            train_texts,
            train_labels,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=True,
        )
        valid_loader = self.prepare_data(
            valid_texts,
            valid_labels,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=False,
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        best_state_dict = None
        best_val_loss = float("inf")

        for epoch in range(epochs):
            train_loss, train_accuracy = self._run_epoch(
                train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                training=True,
            )
            val_loss, val_accuracy = self._run_epoch(valid_loader, training=False)

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            self.model.to(self.device)

        return history

    def _run_epoch(self, data_loader, optimizer=None, scheduler=None, training=False):
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for batch in data_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            if training:
                optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                logits = outputs.logits

                if training:
                    loss.backward()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

            predictions = torch.argmax(logits, dim=1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)

        average_loss = total_loss / max(total_examples, 1)
        accuracy = total_correct / max(total_examples, 1)
        return average_loss, accuracy

    def predict(self, texts, batch_size=16, max_length=256):
        probabilities = self.predict_proba(texts, batch_size=batch_size, max_length=max_length)
        return np.argmax(probabilities, axis=1)

    def predict_labels(self, texts, batch_size=16, max_length=256):
        predictions = self.predict(texts, batch_size=batch_size, max_length=max_length)
        if self.label_encoder is None:
            return predictions
        return self.label_encoder.inverse_transform(predictions)

    def predict_proba(self, texts, batch_size=16, max_length=256):
        data_loader = self.prepare_data(
            texts,
            labels=None,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=False,
        )

        self.model.eval()
        all_probabilities = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                probabilities = torch.softmax(outputs.logits, dim=1)
                all_probabilities.append(probabilities.cpu().numpy())

        return np.vstack(all_probabilities)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path, label_encoder_path=None):
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model_name = path
        self.label_encoder = self._load_label_encoder(label_encoder_path, path)
        self.model.to(self.device)
