"""Transformer-based features: dense vectors that capture the MEANING / deeper
language patterns of a text.

Primary backend : DistilBERT (mean-pooled embeddings) + GPT-2 perplexity
                  (AI text is usually 'less surprising' to a language model => low perplexity).
Fallback backend: TF-IDF + SVD (used only if torch/transformers or the model
                  weights are unavailable, e.g. no internet)."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class TransformerFeatures:
    def __init__(self, bert_name="distilbert-base-uncased", lm_name="gpt2", n_fallback=32, force_fallback=False):
        self.backend = "fallback"
        self.n_fallback = n_fallback
        if not force_fallback:
            try:
                import torch
                from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
                self.torch = torch
                self.tok = AutoTokenizer.from_pretrained(bert_name)
                self.bert = AutoModel.from_pretrained(bert_name).eval()
                self.lm_tok = AutoTokenizer.from_pretrained(lm_name)
                self.lm = AutoModelForCausalLM.from_pretrained(lm_name).eval()
                self.backend = "transformer"
            except Exception as e:  # no torch / no internet
                print(f"[info] Transformer models unavailable ({type(e).__name__}); using TF-IDF fallback.")

    def fit(self, texts):
        if self.backend == "fallback":
            self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            X = self.tfidf.fit_transform(texts)
            k = min(self.n_fallback, X.shape[1] - 1)
            self.svd = TruncatedSVD(n_components=k, random_state=42).fit(X)
        return self

    def _perplexity(self, text):
        torch = self.torch
        enc = self.lm_tok(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            loss = self.lm(**enc, labels=enc["input_ids"]).loss
        return float(torch.exp(loss))

    def transform(self, texts):
        if self.backend == "fallback":
            return self.svd.transform(self.tfidf.transform(texts))
        torch = self.torch
        out = []
        for t in texts:
            enc = self.tok(t, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                h = self.bert(**enc).last_hidden_state.mean(dim=1).squeeze(0).numpy()
            out.append(np.append(h, np.log(self._perplexity(t) + 1e-6)))
        return np.vstack(out)
