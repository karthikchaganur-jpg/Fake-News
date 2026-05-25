import re
import sqlite3
import csv
from pathlib import Path
from datetime import datetime

DB_PATH  = Path("news_dataset/news_articles.db")
OUT_PATH = Path("news_dataset/preprocessed.csv")

# ─── STOP WORDS ──────────────────────────────────────────────────────────────
# Comprehensive English stop word list (no NLTK download needed)
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

# ─── RULE-BASED LEMMATIZER ────────────────────────────────────────────────────
# Covers the most common English inflections without any external download.
IRREGULAR = {
    "are":"be","is":"be","was":"be","were":"be","been":"be","being":"be",
    "has":"have","had":"have","having":"have",
    "does":"do","did":"do","doing":"do","done":"do",
    "went":"go","going":"go","goes":"go","gone":"go",
    "said":"say","says":"say","saying":"say",
    "made":"make","makes":"make","making":"make",
    "took":"take","takes":"take","taking":"take","taken":"take",
    "came":"come","comes":"come","coming":"come",
    "knew":"know","knows":"know","knowing":"know","known":"know",
    "found":"find","finds":"find","finding":"find",
    "gave":"give","gives":"give","giving":"give","given":"give",
    "saw":"see","sees":"see","seen":"see","seeing":"see",
    "thought":"think","thinks":"think","thinking":"think",
    "told":"tell","tells":"tell","telling":"tell",
    "ran":"run","runs":"run","running":"run",
    "brought":"bring","brings":"bring","bringing":"bring",
    "bought":"buy","buys":"buy","buying":"buy",
    "felt":"feel","feels":"feel","feeling":"feel",
    "left":"leave","leaves":"leave","leaving":"leave",
    "children":"child","men":"man","women":"woman","people":"person",
    "lives":"life","knives":"knife","wives":"wife","wolves":"wolf",
}

def lemmatize(word):
    """Rule-based lemmatizer — handles common English inflections."""
    if word in IRREGULAR:
        return IRREGULAR[word]
    # -ing → remove if word is long enough
    if word.endswith("ing") and len(word) > 6:
        stem = word[:-3]
        if stem.endswith(stem[-1]) and len(stem) > 3:   # running → run
            return stem[:-1]
        return stem
    # -ed → base
    if word.endswith("ed") and len(word) > 4:
        stem = word[:-2]
        if stem.endswith(stem[-1]) and len(stem) > 3:   # stopped → stop
            return stem[:-1]
        return stem
    # -ies → y  (studies → study)
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    # -es → e or base  (makes → make, pushes → push)
    if word.endswith("es") and len(word) > 4:
        if word[-3] in "sxzo" or word[-4:-2] == "ch":
            return word[:-2]
        return word[:-1]
    # -s → base  (runs → run)
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word

# ─── PREPROCESSING PIPELINE ──────────────────────────────────────────────────

def preprocess(title, description, content=None):
    """
    Full NLP preprocessing as specified in project PPT:
    1. Combine title + description + content
    2. Lowercase
    3. Punctuation removal
    4. Tokenization
    5. Stop-word removal
    6. Lemmatization
    Returns cleaned text string.
    """
    # 1. Combine fields
    raw = " ".join(filter(None, [title, description, content]))

    # 2. Lowercase
    text = raw.lower()

    # 3. Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # 4. Remove punctuation & special characters (keep letters and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)

    # 5. Tokenize (split on whitespace)
    tokens = text.split()

    # 6. Remove stop words & very short tokens
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    # 7. Lemmatize
    tokens = [lemmatize(t) for t in tokens]

    return " ".join(tokens)

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    if not DB_PATH.exists():
        print("ERROR: news_dataset/news_articles.db not found.")
        print("Run step1_collect.py first.")
        return

    conn = sqlite3.connect(DB_PATH)

    # Load only labeled articles (0=real, 1=fake)
    rows = conn.execute("""
        SELECT id, source_name, label, title, description, content, url, published_at
        FROM articles
        WHERE label IN (0, 1)
        ORDER BY label, source_name
    """).fetchall()
    conn.close()

    if not rows:
        print("No labeled articles found in database.")
        print("Run step1_collect.py first — it auto-labels RSS articles.")
        return

    print(f"\nPreprocessing {len(rows):,} labeled articles...")
    print("Steps: lowercase → remove punctuation → tokenize → remove stop words → lemmatize\n")

    processed = []
    real_count = 0
    fake_count = 0

    for i, (art_id, source, label, title, desc, content, url, pub) in enumerate(rows):
        cleaned = preprocess(title, desc, content)
        if not cleaned.strip():
            continue  # skip empty articles after cleaning

        processed.append({
            "id":             art_id,
            "source_name":    source,
            "label":          label,
            "original_title": title or "",
            "cleaned_text":   cleaned,
            "url":            url or "",
            "published_at":   pub or "",
            "token_count":    len(cleaned.split()),
        })

        if label == 0:
            real_count += 1
        else:
            fake_count += 1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1:,} / {len(rows):,} articles...")

    # Write output CSV
    if processed:
        fieldnames = list(processed[0].keys())
        with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(processed)

    print(f"\n{'='*50}")
    print(f"  PREPROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"  Total processed : {len(processed):,}")
    print(f"  Real  (label=0) : {real_count:,}")
    print(f"  Fake  (label=1) : {fake_count:,}")
    if real_count + fake_count > 0:
        bal = fake_count / (real_count + fake_count) * 100
        print(f"  Class balance   : {bal:.1f}% fake")
    print(f"  Avg token count : {sum(r['token_count'] for r in processed)//max(len(processed),1)}")
    print(f"  Saved to        : {OUT_PATH}")
    print(f"{'='*50}")
    print("\nRun step3_train.py next.")

if __name__ == "__main__":
    main()
