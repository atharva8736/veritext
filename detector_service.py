"""Loads the three detectors (stylometric / transformer / fusion) and turns a text into
everything the website needs: verdict, scores, highlighted sentences, feature report."""
import os, re, zipfile, joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from src.fusion_model import FusionDetector
from src.stylometric import extract, extract_many, FEATURE_NAMES, _sentences

DATA = "data/sample_dataset.csv"
FORCE_FALLBACK = os.environ.get("VERITEXT_FALLBACK", "0") == "1"
MODES = ["stylometric", "transformer", "fusion"]
FEAT_LABELS = {"avg_sentence_len": "Avg sentence length", "burstiness": "Burstiness", "type_token_ratio": "Vocabulary richness",
               "transition_ratio": "Formal linking words", "contraction_ratio": "Contractions",
               "first_person_ratio": "First-person words", "repeated_bigram_ratio": "Repeated phrases"}
AI_MSG = {"avg_sentence_len": ("Very even sentence length", "Sentences are long and regular."),
          "burstiness": ("Very uniform rhythm", "Little variation between sentences."),
          "type_token_ratio": ("Narrow vocabulary", "Words are repeated more than usual."),
          "transition_ratio": ("Many formal linking words", "Moreover, Furthermore, In conclusion."),
          "contraction_ratio": ("No contractions", "Humans usually write don't, it's, I'm."),
          "first_person_ratio": ("No personal voice", "Few words like I, my, we."),
          "repeated_bigram_ratio": ("Repeated phrasing", "Similar word pairs come back often.")}
HUM_MSG = {"avg_sentence_len": ("Varied sentence length", "Short and long sentences are mixed."),
           "burstiness": ("Natural rhythm", "Sentence length changes a lot."),
           "type_token_ratio": ("Rich vocabulary", "Many different words are used."),
           "transition_ratio": ("Few formal linking words", "Reads less like a formal essay."),
           "contraction_ratio": ("Uses contractions", "Casual forms like don't and it's."),
           "first_person_ratio": ("Personal voice", "Words like I and my appear."),
           "repeated_bigram_ratio": ("Little repetition", "Phrases are not reused.")}

class Service:
    def __init__(self):
        df = pd.read_csv(DATA)
        Xtr, _, ytr, _ = train_test_split(df.text.tolist(), df.label.values, test_size=0.25, stratify=df.label, random_state=42)
        self.models = {}
        for m in MODES:
            path = f"models/{m}.joblib"
            if os.path.exists(path):
                try: self.models[m] = joblib.load(path); continue
                except Exception: pass
            self.models[m] = FusionDetector(mode=m, force_fallback=FORCE_FALLBACK).fit(Xtr, ytr)
            joblib.dump(self.models[m], path)
        S = extract_many(df.text.tolist())
        self.hum_mean = S[df.label.values == 0].mean(axis=0)
        self.ai_mean = S[df.label.values == 1].mean(axis=0)
        self.backend = self.models["fusion"].tf.backend

    def p_ai(self, text, mode="fusion"):
        return float(self.models[mode].predict_proba([text])[0])

    def analyze(self, text, mode="fusion"):
        p = self.p_ai(text, mode)
        scores = {m: round(self.p_ai(text, m) * 100) for m in MODES}
        sents = _sentences(text) or [text]
        segs = [{"text": s, "ai": self.p_ai(s, "fusion") >= 0.5} for s in sents]
        vals = extract(text)
        feats = []
        for i, name in enumerate(FEATURE_NAMES):
            if name not in FEAT_LABELS: continue
            h, a = self.hum_mean[i], self.ai_mean[i]
            pos = 0.5 if a == h else float(np.clip((vals[i] - h) / (a - h), 0, 1))
            feats.append({"key": name, "label": FEAT_LABELS[name], "value": f"{vals[i]:.2f}", "pos": round(pos * 100)})
        ranked = sorted(feats, key=lambda f: abs(f["pos"] - 50), reverse=True)[:3]
        signals = [{"ai": f["pos"] >= 50, "title": (AI_MSG if f["pos"] >= 50 else HUM_MSG)[f["key"]][0],
                    "desc": (AI_MSG if f["pos"] >= 50 else HUM_MSG)[f["key"]][1]} for f in ranked]
        return {"p_ai": p, "scores": scores, "segments": segs, "features": feats, "signals": signals,
                "words": len(re.findall(r"\S+", text)), "backend": self.backend, "mode": mode}

def verdict(p, thr=0.5):
    if p >= thr + 0.15: return "ai"
    if p < thr - 0.15: return "hu"
    return "mix"

def read_upload(file):
    name = (file.filename or "").lower()
    data = file.read()
    if name.endswith(".docx"):
        import io
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            xml = z.read("word/document.xml").decode("utf8", "ignore")
        xml = re.sub(r"</w:p>", "\n", xml)
        return re.sub(r"<[^>]+>", "", xml).strip()
    return data.decode("utf8", "ignore").strip()
