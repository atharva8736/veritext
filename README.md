# AI-Generated Text Detection Using Stylometric and Transformer-Based Feature Fusion
Domain: Generative AI & NLP  |  Student: Athrav Singh

## What is this project? (one line)
A program that reads a piece of text and tells you whether it was written by a **human** or by an **AI (like ChatGPT)**.

## Why is it needed?
AI tools can write essays and articles that look human. Teachers, journalists and companies need a way to check authenticity.

## How it works (3 steps)
1. **Stylometric features** (`src/stylometric.py`) - measure the *writing style*: average sentence length, variation in
   sentence length ("burstiness"), vocabulary richness, punctuation, contractions ("don't"), first-person words ("I", "my"),
   formal linking words ("Moreover", "Furthermore"), repeated phrases.
   AI text is smooth, uniform and formal; human text is irregular and personal.
2. **Transformer features** (`src/transformer_features.py`) - a pretrained deep-learning model (DistilBERT) turns the text
   into a vector that captures its *meaning and language patterns*; GPT-2 *perplexity* measures how "predictable" the text is
   (AI text is usually more predictable). If the models can't be downloaded, a TF-IDF fallback is used.
3. **Feature fusion** (`src/fusion_model.py`) - both feature sets are scaled and **joined into one long vector**; a classifier
   (Logistic Regression / Random Forest) predicts Human or AI with a probability.

Text -> [Stylometric features] + [Transformer features] -> FUSION -> Classifier -> Human / AI (+ probability)

## Files
- `make_dataset.py` - creates the demo dataset `data/sample_dataset.csv` (600 synthetic samples)
- `train.py` - trains and compares stylometric-only vs transformer-only vs fusion
- `predict.py` - checks any new text
- `src/` - the three modules above

## How to run
    pip install -r requirements.txt
    python make_dataset.py
    python train.py            # add --fallback if you have no internet / no torch
    python predict.py "Paste any text here"

## Note on the dataset
The included dataset is a small synthetic demo, so scores look near-perfect. For real results use a real corpus
(HC3, GPT-wiki-intro, Kaggle "LLM - Detect AI Generated Text") as a CSV with columns `text,label` (0=human, 1=AI).

## Website (Veritext)
A Flask website with the same design as the app mock-ups: splash, sign in / create account, dashboard,
new analysis (paste or upload .txt/.docx), result with highlighted sentences, detailed report, history and settings.

    pip install -r requirements.txt
    python make_dataset.py        # only if data/sample_dataset.csv is missing
    python app.py                 # then open http://127.0.0.1:5000

Demo login: athrav@demo.com / demo1234  (or create your own account).
The first start trains the three models (stylometric, transformer, fusion) and caches them in `models/`.
With torch + internet the first start also downloads DistilBERT and GPT-2 and takes a few minutes.
No internet / no torch?  Run with  VERITEXT_FALLBACK=1 python app.py   (Windows: set VERITEXT_FALLBACK=1  then  python app.py).

## Folder guide
- `app.py`, `templates/`, `static/` - the Flask website (full project, with transformer + fusion)
- `src/` - stylometric features, transformer features, fusion model
- `online/index.html` - single-page browser version (runs the stylometric model, no server needed)
- `samples/` - one AI-written and one human-style sample text (PDF and TXT)
