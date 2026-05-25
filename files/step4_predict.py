import argparse
import csv
import re
import sys
import time
import joblib
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests

DATA_DIR     = Path("news_dataset")
PRED_CSV     = DATA_DIR / "predictions.csv"
MODEL_PATH   = DATA_DIR / "best_model.pkl"
TFIDF_PATH   = DATA_DIR / "tfidf_vectorizer.pkl"

# ─── SAME STOP WORDS & LEMMATIZER AS STEP 2 ──────────────────────────────────
STOP_WORDS = {
    "a","about","above","after","again","against","all","am","an","and","any",
    "are","aren't","as","at","be","because","been","before","being","below",
    "between","both","but","by","can't","cannot","could","couldn't","did",
    "didn't","do","does","doesn't","doing","don't","down","during","each","few",
    "for","from","further","get","got","had","hadn't","has","hasn't","have",
    "haven't","having","he","he'd","he'll","he's","her","here","here's","hers",
    "herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've",
    "if","in","into","is","isn't","it","it's","its","itself","let's","me","more",
    "most","mustn't","my","myself","no","nor","not","of","off","on","once","only",
    "or","other","ought","our","ours","ourselves","out","over","own","same","shan't",
    "she","she'd","she'll","she's","should","shouldn't","so","some","such","than",
    "that","that's","the","their","theirs","them","themselves","then","there",
    "there's","these","they","they'd","they'll","they're","they've","this","those",
    "through","to","too","under","until","up","very","was","wasn't","we","we'd",
    "we'll","we're","we've","were","weren't","what","what's","when","when's",
    "where","where's","which","while","who","who's","whom","why","why's","will",
    "with","won't","would","wouldn't","you","you'd","you'll","you're","you've",
    "your","yours","yourself","yourselves","said","says","say","also","just",
    "one","two","three","new","now","last","first","like","get","got","make",
    "know","take","see","come","think","look","want","use","find","give","tell",
    "may","might","us","its","mr","ms","mrs","dr","st","year","years","time",
    "day","days","week","weeks","month","months","people","report","reports",
    "reuters","bbc","guardian","according","told","news"
}

IRREGULAR = {
    "are":"be","is":"be","was":"be","were":"be","been":"be",
    "has":"have","had":"have","does":"do","did":"do","done":"do",
    "went":"go","said":"say","made":"make","took":"take","came":"come",
    "knew":"know","found":"find","gave":"give","saw":"see","told":"tell",
    "ran":"run","brought":"bring","bought":"buy","felt":"feel","left":"leave",
}

def lemmatize(word):
    if word in IRREGULAR:
        return IRREGULAR[word]
    if word.endswith("ing") and len(word) > 6:
        stem = word[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            return stem[:-1]
        return stem
    if word.endswith("ed") and len(word) > 4:
        stem = word[:-2]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            return stem[:-1]
        return stem
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 4:
        return word[:-1]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word

def preprocess(title, description="", content=""):
    raw   = " ".join(filter(None, [title, description, content]))
    text  = raw.lower()
    text  = re.sub(r'https?://\S+', '', text)
    text  = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [lemmatize(t) for t in text.split()
              if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)

# ─── LIVE RSS FEEDS FOR PREDICTION ───────────────────────────────────────────
PREDICT_FEEDS = [
    ("reuters_world",  "https://feeds.reuters.com/reuters/worldNews"),
    ("bbc_world",      "http://feeds.bbci.co.uk/news/world/rss.xml"),
    ("guardian_world", "https://www.theguardian.com/world/rss"),
    ("hindu_national", "https://www.thehindu.com/news/national/?service=rss"),
    ("ndtv_top",       "https://feeds.feedburner.com/ndtvnews-top-stories"),
    ("aljazeera",      "https://www.aljazeera.com/xml/rss/all.xml"),
    ("theonion",       "https://www.theonion.com/rss"),
    ("babylonbee",     "https://babylonbee.com/feed"),
]

def fetch_live_articles():
    articles = []
    for source, url in PREDICT_FEEDS:
        print(f"  Fetching {source}...")
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:15]:
                title = re.sub(r'<[^>]+>', '', getattr(entry, 'title', '') or '')
                desc  = re.sub(r'<[^>]+>', '', getattr(entry, 'summary', '') or '')
                if title:
                    articles.append({
                        "source": source,
                        "title": title.strip(),
                        "description": desc.strip()[:500],
                        "url": getattr(entry, 'link', ''),
                        "published": getattr(entry, 'published', ''),
                    })
        except Exception as e:
            print(f"  ERROR {source}: {e}")
        time.sleep(0.5)
    return articles

# ─── PREDICT ─────────────────────────────────────────────────────────────────
def predict_articles(articles, model, vectorizer):
    results = []
    for a in articles:
        cleaned = preprocess(a["title"], a.get("description", ""))
        if not cleaned.strip():
            continue
        vec        = vectorizer.transform([cleaned])
        pred       = model.predict(vec)[0]
        proba      = model.predict_proba(vec)[0]
        prob_fake  = proba[1]
        prob_real  = proba[0]
        confidence = max(prob_fake, prob_real)
        label      = "FAKE" if pred == 1 else "REAL"

        results.append({
            **a,
            "prediction":  label,
            "prob_fake":   round(prob_fake, 4),
            "prob_real":   round(prob_real, 4),
            "confidence":  round(confidence, 4),
            "cleaned_text": cleaned,
        })

    # Sort by prob_fake descending
    results.sort(key=lambda x: x["prob_fake"], reverse=True)
    return results

# ─── DISPLAY ─────────────────────────────────────────────────────────────────
def display_results(results):
    fake_results = [r for r in results if r["prediction"] == "FAKE"]
    real_results = [r for r in results if r["prediction"] == "REAL"]

    print(f"\n{'='*60}")
    print(f"  PREDICTION RESULTS — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")
    print(f"  Total classified : {len(results)}")
    print(f"  Predicted REAL   : {len(real_results)}")
    print(f"  Predicted FAKE   : {len(fake_results)}")
    print(f"{'='*60}")

    if fake_results:
        print(f"\n  ⚠  TOP FAKE / SATIRICAL ARTICLES (by probability)")
        print(f"{'─'*60}")
        for r in fake_results[:10]:
            bar_len = int(r["prob_fake"] * 30)
            bar     = "█" * bar_len + "░" * (30 - bar_len)
            print(f"\n  Source : {r['source']}")
            print(f"  Title  : {r['title'][:75]}{'...' if len(r['title'])>75 else ''}")
            print(f"  P(fake): {bar}  {r['prob_fake']*100:.1f}%")

    if real_results:
        print(f"\n\n  ✓  TOP REAL ARTICLES (most confident)")
        print(f"{'─'*60}")
        for r in real_results[-10:]:
            bar_len = int(r["prob_real"] * 30)
            bar     = "█" * bar_len + "░" * (30 - bar_len)
            print(f"\n  Source : {r['source']}")
            print(f"  Title  : {r['title'][:75]}{'...' if len(r['title'])>75 else ''}")
            print(f"  P(real): {bar}  {r['prob_real']*100:.1f}%")

    print(f"\n{'='*60}")

def save_predictions(results):
    if not results:
        return
    fields = ["source","title","prediction","prob_fake","prob_real",
              "confidence","url","published","description"]
    with open(PRED_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"  Predictions saved → {PRED_CSV}")

# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",    default="", help="Classify a single headline")
    parser.add_argument("--newsapi", default="", help="NewsAPI key for extra articles")
    args = parser.parse_args()

    # Load model + vectorizer
    if not MODEL_PATH.exists() or not TFIDF_PATH.exists():
        print("ERROR: Model not found. Run step3_train.py first.")
        return

    print("Loading model and vectorizer...")
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(TFIDF_PATH)
    print("Model loaded.\n")

    # Single text mode
    if args.text:
        cleaned   = preprocess(args.text)
        vec       = vectorizer.transform([cleaned])
        pred      = model.predict(vec)[0]
        proba     = model.predict_proba(vec)[0]
        print(f"\nInput   : {args.text}")
        print(f"Result  : {'FAKE' if pred==1 else 'REAL'}")
        print(f"P(fake) : {proba[1]*100:.1f}%")
        print(f"P(real) : {proba[0]*100:.1f}%")
        return

    # Live feed mode
    print("Fetching live news articles...")
    articles = fetch_live_articles()
    print(f"Fetched {len(articles)} articles\n")

    print("Classifying...")
    results = predict_articles(articles, model, vectorizer)

    display_results(results)
    save_predictions(results)

if __name__ == "__main__":
    main()
