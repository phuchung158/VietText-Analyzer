import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class BaselineModel:
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), 
                                      min_df=2, max_df=0.8)),
            ('lr', LogisticRegression(max_iter=1000, random_state=42))
        ])
    
    def train(self, texts, labels):
        """Train mô hình"""
        print("Training Baseline (TF-IDF + LR)...")
        self.model.fit(texts, labels)
        print("✓ Training completed")
    
    def predict(self, texts):
        """Dự đoán"""
        return self.model.predict(texts)
    
    def predict_proba(self, texts):
        """Xác suất dự đoán"""
        return self.model.predict_proba(texts)
    
    def save(self, path):
        """Lưu mô hình"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, path):
        """Load mô hình"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
