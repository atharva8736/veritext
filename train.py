"""Train + evaluate. Compares: stylometric-only, transformer-only, and FUSION."""
import argparse, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from src.fusion_model import FusionDetector

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/sample_dataset.csv")
ap.add_argument("--classifier", default="logreg", choices=["logreg", "rf"])
ap.add_argument("--fallback", action="store_true", help="skip HuggingFace models, use TF-IDF fallback")
a = ap.parse_args()

df = pd.read_csv(a.data)
Xtr, Xte, ytr, yte = train_test_split(df.text.tolist(), df.label.values, test_size=0.25, stratify=df.label, random_state=42)
print(f"Train: {len(Xtr)}  Test: {len(Xte)}\n")
print(f"{'Model':<22}{'Acc':>7}{'Prec':>7}{'Rec':>7}{'F1':>7}")
best = None
for mode in ["stylometric", "transformer", "fusion"]:
    m = FusionDetector(mode=mode, classifier=a.classifier, force_fallback=a.fallback).fit(Xtr, ytr)
    p = m.predict(Xte)
    pr, rc, f1, _ = precision_recall_fscore_support(yte, p, average="binary")
    print(f"{mode:<22}{accuracy_score(yte, p):>7.3f}{pr:>7.3f}{rc:>7.3f}{f1:>7.3f}")
    if mode == "fusion":
        best = m; cm = confusion_matrix(yte, p)
print("\nConfusion matrix (fusion) [rows=true Human/AI, cols=pred Human/AI]:\n", cm)
joblib.dump(best, "models/fusion_detector.joblib")
print("\nSaved models/fusion_detector.joblib")
