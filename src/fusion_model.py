"""Feature fusion: stylometric features + transformer features are scaled and
concatenated into ONE vector, then a classifier decides Human vs AI."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .stylometric import extract_many, FEATURE_NAMES
from .transformer_features import TransformerFeatures

LABELS = {0: "Human-written", 1: "AI-generated"}

class FusionDetector:
    def __init__(self, mode="fusion", classifier="logreg", force_fallback=False):
        assert mode in ("stylometric", "transformer", "fusion")
        self.mode = mode
        self.tf = TransformerFeatures(force_fallback=force_fallback)
        self.sty_scaler, self.tf_scaler = StandardScaler(), StandardScaler()
        self.clf = (LogisticRegression(max_iter=2000, C=1.0) if classifier == "logreg"
                    else RandomForestClassifier(n_estimators=300, random_state=42))

    def _features(self, texts, fit=False):
        parts = []
        if self.mode in ("stylometric", "fusion"):
            S = extract_many(texts)
            parts.append(self.sty_scaler.fit_transform(S) if fit else self.sty_scaler.transform(S))
        if self.mode in ("transformer", "fusion"):
            if fit:
                self.tf.fit(texts)
            T = self.tf.transform(texts)
            parts.append(self.tf_scaler.fit_transform(T) if fit else self.tf_scaler.transform(T))
        return np.hstack(parts)

    def fit(self, texts, y):
        self.clf.fit(self._features(texts, fit=True), y)
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self._features(texts))[:, 1]   # P(AI)

    def predict(self, texts):
        return (self.predict_proba(texts) >= 0.5).astype(int)

    def explain_style(self, text):
        """Return stylometric feature values for one text (for the demo)."""
        from .stylometric import extract
        return dict(zip(FEATURE_NAMES, extract(text).round(3)))
