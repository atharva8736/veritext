"""Creates a small DEMO dataset (data/sample_dataset.csv) so the project runs anywhere.
Texts are synthetic: 'AI-style' = formal, balanced, transition-heavy; 'human-style' =
casual, irregular, personal. For real results, replace the CSV with a real corpus
(e.g. HC3, GPT-wiki-intro, or the Kaggle 'LLM - Detect AI Generated Text' data)
using columns: text,label  (0 = human, 1 = AI)."""
import random, pandas as pd
random.seed(7)
topics = {
 "climate change": ["rising global temperatures", "melting polar ice", "carbon emissions", "extreme weather events", "renewable energy adoption"],
 "social media": ["online communication", "teenage mental health", "misinformation", "digital marketing", "personal privacy"],
 "exercise": ["cardiovascular health", "muscle strength", "sleep quality", "stress reduction", "daily routines"],
 "online education": ["flexible learning", "access to resources", "student engagement", "digital literacy", "remote classrooms"],
 "electric cars": ["battery technology", "charging infrastructure", "lower running costs", "air pollution", "government incentives"],
 "artificial intelligence": ["automation", "data privacy", "healthcare diagnostics", "job markets", "ethical responsibility"],
 "healthy eating": ["balanced nutrition", "processed foods", "vitamin intake", "meal planning", "long-term wellbeing"],
 "space exploration": ["scientific discovery", "satellite technology", "international cooperation", "funding priorities", "future colonies"],
}
ai_open = ["It is important to note that {t} plays a crucial role in modern society.",
           "{T} has become an increasingly significant topic in recent years.",
           "In today's rapidly evolving world, {t} offers numerous opportunities and challenges."]
ai_mid = ["Moreover, {a} contributes significantly to the overall outcome.",
          "Furthermore, {a} can have a profound impact on individuals and communities alike.",
          "Additionally, {a} is essential for fostering sustainable and long-term progress.",
          "Consequently, understanding {a} is vital for making informed decisions."]
ai_end = ["In conclusion, a balanced approach to {t} is necessary to ensure positive results for everyone.",
          "Overall, {t} represents both a challenge and an opportunity that requires careful consideration.",
          "Ultimately, addressing {t} effectively will benefit society as a whole."]
hu_open = ["Honestly, I don't really get why everyone's so obsessed with {t}.", "So I was thinking about {t} yesterday and, well, it got weird.",
           "My brother keeps ranting about {t}, and ok, he's got a point.", "I've got mixed feelings about {t}, to be fair."]
hu_mid = ["Like, {a}? Yeah that matters. A lot.", "{A} is kinda the thing nobody talks about at dinner.", "I read something on {a} last week and it stuck with me.",
          "Not sure if I'm right but {a} seems way more complicated than people say.", "Anyway, {a}. Whatever you think, it's not simple!", "We argued about {a} for like an hour lol."]
hu_end = ["Don't know. Maybe I'm wrong?", "Anyway, that's my two cents.", "Can't say I've figured it out yet.", "Guess we'll see what happens."]
formal_leak = ["The data on {a} is widely discussed.", "Researchers have studied {a} extensively."]
casual_leak = ["I mean, {a} is pretty important.", "Honestly {a} matters."]

def make(kind):
    t = random.choice(list(topics)); a = random.sample(topics[t], 3)
    if kind == "ai":
        s = [random.choice(ai_open).format(t=t, T=t.capitalize())]
        s += [random.choice(ai_mid).format(a=x) for x in a[:random.randint(2, 3)]]
        if random.random() < .2: s.insert(1, random.choice(casual_leak).format(a=a[0]))   # label noise
        s.append(random.choice(ai_end).format(t=t))
    else:
        s = [random.choice(hu_open).format(t=t)]
        s += [random.choice(hu_mid).format(a=x, A=x.capitalize()) for x in a[:random.randint(1, 3)]]
        if random.random() < .25: s.insert(1, random.choice(formal_leak).format(a=a[0]))  # label noise
        s.append(random.choice(hu_end))
    return " ".join(s)

rows = [(make("human"), 0) for _ in range(300)] + [(make("ai"), 1) for _ in range(300)]
random.shuffle(rows)
pd.DataFrame(rows, columns=["text", "label"]).to_csv("data/sample_dataset.csv", index=False)
print("Saved data/sample_dataset.csv with", len(rows), "rows")
