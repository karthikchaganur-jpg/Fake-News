import argparse, hashlib, re, sqlite3, joblib
from datetime import datetime, timezone
from pathlib import Path

DB_PATH    = Path("news_dataset/news_articles.db")
MODEL_PATH = Path("news_dataset/best_model.pkl")
TFIDF_PATH = Path("news_dataset/tfidf_vectorizer.pkl")
LOG_PATH   = Path("news_dataset/collection_log.txt")

STOP_WORDS = {
    "a","about","above","after","again","against","all","am","an","and","any",
    "are","as","at","be","because","been","before","being","below","between",
    "both","but","by","cannot","could","did","do","does","doing","down","during",
    "each","few","for","from","further","get","got","had","has","have","having",
    "he","her","here","hers","herself","him","himself","his","how","if","in",
    "into","is","it","its","itself","me","more","most","my","myself","no","nor",
    "not","of","off","on","once","only","or","other","our","ours","ourselves",
    "out","over","own","same","she","should","so","some","such","than","that",
    "the","their","theirs","them","themselves","then","there","these","they",
    "this","those","through","to","too","under","until","up","very","was","we",
    "were","what","when","where","which","while","who","whom","why","will","with",
    "would","you","your","yours","yourself","yourselves","said","says","say",
    "also","just","one","two","three","new","now","last","first","like","make",
    "know","take","see","come","think","look","want","use","find","give","tell",
    "may","might","us","mr","ms","mrs","dr","st","year","years","time","day",
    "days","week","weeks","month","months","people","report","reports",
    "reuters","bbc","guardian","according","told","news"
}

IRREGULAR = {
    "are":"be","is":"be","was":"be","were":"be","been":"be","has":"have",
    "had":"have","does":"do","did":"do","done":"do","went":"go","said":"say",
    "made":"make","took":"take","came":"come","knew":"know","found":"find",
    "gave":"give","saw":"see","told":"tell","ran":"run","brought":"bring",
    "bought":"buy","felt":"feel","left":"leave",
}

def lemmatize(word):
    if word in IRREGULAR: return IRREGULAR[word]
    if word.endswith("ing") and len(word)>6:
        stem=word[:-3]
        return stem[:-1] if len(stem)>2 and stem[-1]==stem[-2] else stem
    if word.endswith("ed") and len(word)>4:
        stem=word[:-2]
        return stem[:-1] if len(stem)>2 and stem[-1]==stem[-2] else stem
    if word.endswith("ies") and len(word)>4: return word[:-3]+"y"
    if word.endswith("es") and len(word)>4: return word[:-1]
    if word.endswith("s") and not word.endswith("ss") and len(word)>3: return word[:-1]
    return word

def preprocess(title, body=""):
    raw=(" ".join(filter(None,[title,body]))).lower()
    raw=re.sub(r'https?://\S+','',raw)
    raw=re.sub(r'[^a-z\s]',' ',raw)
    return " ".join(lemmatize(t) for t in raw.split() if t not in STOP_WORDS and len(t)>2)

def load_model():
    if MODEL_PATH.exists() and TFIDF_PATH.exists():
        return joblib.load(MODEL_PATH), joblib.load(TFIDF_PATH)
    return None, None

def predict(title, body, model, vectorizer):
    vec=vectorizer.transform([preprocess(title,body)])
    pred=model.predict(vec)[0]
    proba=model.predict_proba(vec)[0]
    return int(pred), proba[1], proba[0]

def display_prediction(title, label, prob_fake, prob_real):
    verdict = "FAKE / MISLEADING" if label==1 else "REAL / CREDIBLE"
    emoji   = "WARNING" if label==1 else "OK"
    fake_bar= "█"*int(prob_fake*40)+"░"*(40-int(prob_fake*40))
    real_bar= "█"*int(prob_real*40)+"░"*(40-int(prob_real*40))
    print("\n"+"="*55)
    print("  PREDICTION RESULT")
    print("="*55)
    print(f"  Title   : {title[:60]}{'...' if len(title)>60 else ''}")
    print(f"  Verdict : [{emoji}]  {verdict}")
    print(f"  P(fake) : {fake_bar}  {prob_fake*100:.1f}%")
    print(f"  P(real) : {real_bar}  {prob_real*100:.1f}%")
    print("="*55)
    if   prob_fake>=0.80: print("  Very high chance this is FAKE or misleading.")
    elif prob_fake>=0.60: print("  Likely FAKE — treat with caution.")
    elif prob_real>=0.80: print("  Very likely REAL and credible.")
    elif prob_real>=0.60: print("  Probably REAL, but verify independently.")
    else:                 print("  Model is uncertain — verify independently.")
    print()

SCHEMA="""CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY, collected_at TEXT, source_name TEXT,
    label INTEGER DEFAULT -1, title TEXT, description TEXT,
    content TEXT, url TEXT, published_at TEXT, author TEXT,
    category TEXT, collection_method TEXT,
    word_count INTEGER, char_count INTEGER,
    exclamation_count INTEGER, caps_word_count INTEGER,
    has_question_title INTEGER);"""

def init_db():
    Path("news_dataset").mkdir(exist_ok=True)
    conn=sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    conn.commit()
    return conn

def basic_features(title,body):
    full=(title or "")+" "+(body or "")
    words=full.split()
    return {"word_count":len(words),"char_count":len(full),
            "exclamation_count":full.count("!"),
            "caps_word_count":sum(1 for w in words if re.match(r'^[A-Z]{3,}$',w)),
            "has_question_title":int((title or "").strip().endswith("?"))}

def save_article(conn,title,body,label,source="manual_upload"):
    row={
        "id":hashlib.md5((title+body).encode()).hexdigest(),
        "collected_at":datetime.now(timezone.utc).isoformat(),
        "source_name":source,"label":label,
        "title":title.strip(),"description":body[:1000].strip(),
        "content":body[:5000].strip(),"url":"",
        "published_at":datetime.now(timezone.utc).isoformat(),
        "author":"","category":"manual","collection_method":"manual_upload",
        **basic_features(title,body),
    }
    cols=list(row.keys())
    cur=conn.execute(
        f"INSERT OR IGNORE INTO articles ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
        [row[c] for c in cols])
    conn.commit()
    return cur.rowcount>0

def log(msg):
    with open(LOG_PATH,"a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")

def read_txt_file(filepath):
    path=Path(filepath)
    if not path.exists():
        print(f"ERROR: File not found — {filepath}"); return None,None,None
    lines=path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        print("ERROR: File is empty."); return None,None,None
    title=lines[0].strip()
    source=lines[1].strip() if len(lines)>1 else "manual_upload"
    body="\n".join(lines[2:]).strip() if len(lines)>2 else ""
    if source and len(source)>80:
        body=source+"\n"+body; source="manual_upload"
    return title,source,body

def interactive_mode(conn, model, vectorizer):
    has_model = model is not None
    print("\n"+"="*55)
    print("  ADD & CLASSIFY ARTICLE")
    print("="*55)
    if has_model:
        print("  Model loaded — articles will be AUTO-CLASSIFIED.\n")
    else:
        print("  No model yet — run step1→step2→step3 first.")
        print("  Articles saved as unlabeled for now.\n")

    while True:
        print("─"*55)
        title=input("  Paste headline / title:\n  > ").strip()
        if not title:
            print("  Title cannot be empty."); continue

        source=input("  Source name (press Enter to skip):\n  > ").strip() or "manual_upload"

        print("  Paste article body (press Enter TWICE when done):")
        lines=[]
        while True:
            line=input()
            if line=="" and lines and lines[-1]=="": break
            lines.append(line)
        body="\n".join(lines).strip()

        if has_model:
            label, prob_fake, prob_real = predict(title, body, model, vectorizer)
            display_prediction(title, label, prob_fake, prob_real)
            confirm=input("  Does this prediction look correct? [y/n]: ").strip().lower()
            if confirm=="n":
                override=input("  Enter correct label — 0=real  1=fake: ").strip()
                if override in ("0","1"):
                    label=int(override)
                    print(f"  Label overridden to {'FAKE' if label==1 else 'REAL'}.")
        else:
            label=-1

        verdict="FAKE" if label==1 else "REAL" if label==0 else "UNLABELED"
        if save_article(conn,title,body,label,source):
            log(f"Saved [{verdict}] — {title[:60]}")
            print(f"  Saved to dataset as {verdict}.\n")
        else:
            print("  Already exists in dataset (duplicate).\n")

        if input("  Classify another? [y/n]: ").strip().lower()!="y":
            break

    total=conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    real =conn.execute("SELECT COUNT(*) FROM articles WHERE label=0").fetchone()[0]
    fake =conn.execute("SELECT COUNT(*) FROM articles WHERE label=1").fetchone()[0]
    print(f"\n  Dataset: {total} total — {real} real, {fake} fake.")
    print("="*55)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--title",  default="")
    parser.add_argument("--text",   default="")
    parser.add_argument("--file",   default="")
    parser.add_argument("--source", default="manual_upload")
    args=parser.parse_args()

    conn=init_db()
    model,vectorizer=load_model()

    if model: print("  Model loaded ✓")
    else:     print("  No model found — run step3_train.py first for auto-prediction.")

    if args.file:
        title,source,body=read_txt_file(args.file)
        if not title: return
        source=args.source if args.source!="manual_upload" else source
        if model:
            label,prob_fake,prob_real=predict(title,body,model,vectorizer)
            display_prediction(title,label,prob_fake,prob_real)
        else:
            label=-1
        verdict="FAKE" if label==1 else "REAL" if label==0 else "UNLABELED"
        if save_article(conn,title,body,label,source):
            log(f"Saved [{verdict}] from file — {title[:60]}")
            print(f"  Saved as {verdict}.")
        else:
            print("  Already exists in dataset.")

    elif args.title and args.text:
        if model:
            label,prob_fake,prob_real=predict(args.title,args.text,model,vectorizer)
            display_prediction(args.title,label,prob_fake,prob_real)
        else:
            label=-1
        verdict="FAKE" if label==1 else "REAL" if label==0 else "UNLABELED"
        if save_article(conn,args.title,args.text,label,args.source):
            log(f"Saved [{verdict}] — {args.title[:60]}")
            print(f"  Saved as {verdict}.")
        else:
            print("  Already exists in dataset.")

    else:
        interactive_mode(conn,model,vectorizer)

    conn.close()

if __name__=="__main__":
    main()
