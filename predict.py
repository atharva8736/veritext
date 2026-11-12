"""Usage:  python predict.py "your text here"     or     python predict.py --file essay.txt"""
import sys, argparse, joblib
ap = argparse.ArgumentParser(); ap.add_argument("text", nargs="?"); ap.add_argument("--file")
a = ap.parse_args()
text = open(a.file, encoding="utf-8").read() if a.file else a.text
if not text: sys.exit("Give some text or --file")
m = joblib.load("models/fusion_detector.joblib")
p = float(m.predict_proba([text])[0])
print(f"\nPrediction : {'AI-generated' if p >= .5 else 'Human-written'}")
print(f"P(AI)      : {p:.1%}   P(Human): {1-p:.1%}\n")
print("Key stylometric features:")
for k in ["avg_sentence_len", "burstiness", "type_token_ratio", "transition_ratio", "contraction_ratio", "first_person_ratio"]:
    print(f"  {k:<20}{m.explain_style(text)[k]}")
