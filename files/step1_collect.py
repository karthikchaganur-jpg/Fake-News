import argparse
import csv
import hashlib
import json
import re
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import feedparser
import requests

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("news_dataset")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH  = DATA_DIR / "news_articles.db"
CSV_PATH = DATA_DIR / "news_articles.csv"
LOG_PATH = DATA_DIR / "collection_log.txt"

# Kaggle CSV paths — place True.csv and Fake.csv next to this script
KAGGLE_REAL_CSV = Path(__file__).parent / "True.csv"
KAGGLE_FAKE_CSV = Path(__file__).parent / "Fake.csv"

# How many articles to load from each Kaggle CSV (500+500 = 1000 total)
KAGGLE_LIMIT_PER_CLASS = 500

DELAY        = 1.5
MAX_PER_FEED = 200

# ─── RSS SOURCES ─────────────────────────────────────────────────────────────
RSS_SOURCES = {
    # REAL label=0
    "reuters_world":      ("https://feeds.reuters.com/reuters/worldNews",             0),
    "reuters_tech":       ("https://feeds.reuters.com/reuters/technologyNews",        0),
    "reuters_science":    ("https://feeds.reuters.com/reuters/scienceNews",           0),
    "reuters_health":     ("https://feeds.reuters.com/reuters/healthNews",            0),
    "reuters_politics":   ("https://feeds.reuters.com/reuters/politicsNews",          0),
    "bbc_world":          ("http://feeds.bbci.co.uk/news/world/rss.xml",              0),
    "bbc_tech":           ("http://feeds.bbci.co.uk/news/technology/rss.xml",         0),
    "bbc_health":         ("http://feeds.bbci.co.uk/news/health/rss.xml",             0),
    "bbc_science":        ("http://feeds.bbci.co.uk/news/science_and_environment/rss.xml", 0),
    "guardian_world":     ("https://www.theguardian.com/world/rss",                   0),
    "guardian_tech":      ("https://www.theguardian.com/technology/rss",              0),
    "guardian_science":   ("https://www.theguardian.com/science/rss",                0),
    "guardian_politics":  ("https://www.theguardian.com/politics/rss",               0),
    "hindu_national":     ("https://www.thehindu.com/news/national/?service=rss",    0),
    "hindu_science":      ("https://www.thehindu.com/sci-tech/science/?service=rss", 0),
    "hindu_international":("https://www.thehindu.com/news/international/?service=rss", 0),
    "ie_india":           ("https://indianexpress.com/feed/",                         0),
    "ndtv_top":           ("https://feeds.feedburner.com/ndtvnews-top-stories",       0),
    "livemint":           ("https://www.livemint.com/rss/news",                       0),
    "toi_top":            ("https://timesofindia.indiatimes.com/rssfeedstopstories.cms", 0),
    "npr_news":           ("https://feeds.npr.org/1001/rss.xml",                     0),
    "pbs_world":          ("https://www.pbs.org/newshour/feeds/rss/world",            0),
    "nasa":               ("https://www.nasa.gov/rss/dyn/breaking_news.rss",          0),
    "who":                ("https://www.who.int/rss-feeds/news-english.xml",          0),
    "ap_topnews":         ("https://rsshub.app/apnews/topics/apf-topnews",            0),
    "aljazeera":          ("https://www.aljazeera.com/xml/rss/all.xml",               0),
    "abc_news":           ("https://feeds.abcnews.com/abcnews/topstories",            0),
    "cbs_news":           ("https://www.cbsnews.com/latest/rss/main",                 0),
    "science_daily":      ("https://www.sciencedaily.com/rss/all.xml",                0),
    # FAKE label=1
    "theonion":           ("https://www.theonion.com/rss",                            1),
    "babylonbee":         ("https://babylonbee.com/feed",                             1),
    "waterfordwhispers":  ("https://waterfordwhispersnews.com/feed/",                 1),
    "nationalreport":     ("https://nationalreport.net/feed/",                        1),
    "worldnewsdailyreport":("https://worldnewsdailyreport.com/feed/",                1),
    "empirenews":         ("https://empirenews.net/feed/",                            1),
    "thedailymash":       ("https://www.thedailymash.co.uk/feed",                     1),
    "newsthump":          ("https://newsthump.com/feed/",                             1),
    "thespoof":           ("https://www.thespoof.com/rss.php",                        1),
}

WAYBACK_FEEDS = {
    "wb_reuters_world":  ("https://feeds.reuters.com/reuters/worldNews",              0),
    "wb_reuters_tech":   ("https://feeds.reuters.com/reuters/technologyNews",         0),
    "wb_reuters_health": ("https://feeds.reuters.com/reuters/healthNews",             0),
    "wb_bbc_world":      ("http://feeds.bbci.co.uk/news/world/rss.xml",               0),
    "wb_bbc_health":     ("http://feeds.bbci.co.uk/news/health/rss.xml",              0),
    "wb_guardian_world": ("https://www.theguardian.com/world/rss",                    0),
    "wb_guardian_tech":  ("https://www.theguardian.com/technology/rss",               0),
    "wb_hindu":          ("https://www.thehindu.com/news/national/?service=rss",     0),
    "wb_ndtv":           ("https://feeds.feedburner.com/ndtvnews-top-stories",        0),
    "wb_npr":            ("https://feeds.npr.org/1001/rss.xml",                      0),
    "wb_aljazeera":      ("https://www.aljazeera.com/xml/rss/all.xml",                0),
    "wb_theonion":       ("https://www.theonion.com/rss",                             1),
    "wb_babylonbee":     ("https://babylonbee.com/feed",                              1),
    "wb_waterford":      ("https://waterfordwhispersnews.com/feed/",                  1),
    "wb_thedailymash":   ("https://www.thedailymash.co.uk/feed",                      1),
    "wb_newsthump":      ("https://newsthump.com/feed/",                              1),
}

# ─── SCHEMA ──────────────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY, collected_at TEXT NOT NULL,
    source_name TEXT NOT NULL, label INTEGER NOT NULL DEFAULT -1,
    title TEXT, description TEXT, content TEXT, url TEXT,
    published_at TEXT, author TEXT, category TEXT, collection_method TEXT,
    word_count INTEGER, char_count INTEGER,
    exclamation_count INTEGER, caps_word_count INTEGER, has_question_title INTEGER
);
CREATE INDEX IF NOT EXISTS idx_label  ON articles(label);
CREATE INDEX IF NOT EXISTS idx_source ON articles(source_name);
CREATE INDEX IF NOT EXISTS idx_date   ON articles(published_at);
"""

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def make_id(url, title):
    return hashlib.md5(((url or "") + (title or "")).encode()).hexdigest()

def strip_html(text):
    return re.sub(r'<[^>]+>', '', text or '').strip()

def basic_features(title, desc):
    text  = (title or "") + " " + (desc or "")
    words = text.split()
    return {
        "word_count":         len(words),
        "char_count":         len(text),
        "exclamation_count":  text.count("!"),
        "caps_word_count":    sum(1 for w in words if re.match(r'^[A-Z]{3,}$', w)),
        "has_question_title": int((title or "").strip().endswith("?")),
    }

def parse_date(date_str):
    if not date_str:
        return ""
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S GMT",
                "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(date_str.strip(), fmt).isoformat()
        except Exception:
            pass
    return date_str

def is_within_window(date_str, date_from):
    if not date_str:
        return True
    try:
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt >= date_from
    except Exception:
        return True

def init_db():
    conn = sqlite3.connect(DB_PATH)
    for stmt in SCHEMA.strip().split(";"):
        s = stmt.strip()
        if s:
            conn.execute(s)
    conn.commit()
    return conn

def insert(conn, row):
    cols = list(row.keys())
    sql  = (f"INSERT OR IGNORE INTO articles ({','.join(cols)}) "
            f"VALUES ({','.join('?'*len(cols))})")
    cur  = conn.execute(sql, [row[c] for c in cols])
    conn.commit()
    return cur.rowcount > 0

def export_csv(conn):
    cur  = conn.execute("SELECT * FROM articles ORDER BY collected_at DESC")
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([cols] + list(rows))
    log(f"Exported {len(rows):,} rows to {CSV_PATH}")


# ─── COLLECTOR 0: KAGGLE DATASET ─────────────────────────────────────────────
def collect_kaggle(conn):
    """
    Loads up to KAGGLE_LIMIT_PER_CLASS articles from each of:
      True.csv  — real news (label=0)
      Fake.csv  — fake news (label=1)

    These files must be placed in the same directory as this script.
    Download from:
      https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

    CSV columns expected: title, text, subject, date
    (The dataset also has a 'subject' column which is used as category.)
    """
    log("=" * 55)
    log("STEP: Kaggle Fake & Real News Dataset")
    log(f"  Loading up to {KAGGLE_LIMIT_PER_CLASS} real + {KAGGLE_LIMIT_PER_CLASS} fake articles")
    log("=" * 55)

    sources = [
        (KAGGLE_REAL_CSV, 0, "kaggle_real"),
        (KAGGLE_FAKE_CSV, 1, "kaggle_fake"),
    ]

    total = 0
    now   = datetime.now(timezone.utc).isoformat()

    for csv_path, label, source_name in sources:
        tag = "[REAL]" if label == 0 else "[FAKE]"

        if not csv_path.exists():
            log(f"  WARNING: {csv_path.name} not found — skipping {tag}")
            log(f"  Download from https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
            log(f"  and place {csv_path.name} next to step1_collect.py")
            continue

        count    = 0
        skipped  = 0

        try:
            with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)

                for row_num, raw in enumerate(reader):
                    if count >= KAGGLE_LIMIT_PER_CLASS:
                        break

                    # Kaggle dataset columns: title, text, subject, date
                    title   = (raw.get("title") or "").strip()
                    body    = (raw.get("text")  or "").strip()
                    subject = (raw.get("subject") or "").strip()
                    date_s  = (raw.get("date")  or "").strip()

                    if not title:
                        skipped += 1
                        continue

                    # Use first 5000 chars of body to keep DB lean
                    body_trimmed = body[:5000]

                    row = {
                        "id":                make_id(title, body[:200]),
                        "collected_at":      now,
                        "source_name":       source_name,
                        "label":             label,
                        "title":             title,
                        "description":       body_trimmed[:1000],
                        "content":           body_trimmed,
                        "url":               "",
                        "published_at":      parse_date(date_s) or now,
                        "author":            "kaggle",
                        "category":          subject or "kaggle",
                        "collection_method": "kaggle_csv",
                        **basic_features(title, body_trimmed),
                    }

                    if insert(conn, row):
                        count += 1

        except Exception as e:
            log(f"  ERROR reading {csv_path.name}: {e}")
            continue

        label_str = "real" if label == 0 else "fake"
        log(f"  {tag} kaggle_{label_str}: +{count} articles  (skipped {skipped} empty rows)")
        total += count

    log(f"  Kaggle total: {total} new articles\n")
    return total


# ─── COLLECTOR 1: LIVE RSS ────────────────────────────────────────────────────
def collect_rss(conn, date_from):
    log("=" * 55)
    log("STEP: Live RSS feeds")
    log("=" * 55)
    total = 0
    for name, (url, label) in RSS_SOURCES.items():
        tag = "[REAL]" if label == 0 else "[FAKE]"
        log(f"  {tag} {name} ...")
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            log(f"    ERROR: {e}"); continue

        count = 0
        for entry in feed.entries[:MAX_PER_FEED]:
            title  = strip_html(getattr(entry, 'title', ''))
            desc   = strip_html(getattr(entry, 'summary', '') or getattr(entry, 'description', ''))
            url_   = getattr(entry, 'link', '')
            pub    = parse_date(getattr(entry, 'published', '') or getattr(entry, 'updated', ''))
            author = getattr(entry, 'author', '')
            if not title or not is_within_window(pub, date_from):
                continue
            row = {
                "id": make_id(url_, title), "collected_at": datetime.now(timezone.utc).isoformat(),
                "source_name": name, "label": label, "title": title,
                "description": desc[:1000], "content": None, "url": url_,
                "published_at": pub, "author": author, "category": None,
                "collection_method": "rss", **basic_features(title, desc),
            }
            if insert(conn, row):
                count += 1

        log(f"    +{count} new articles")
        total += count
        time.sleep(DELAY)

    log(f"RSS done: {total:,} new articles\n")
    return total


# ─── COLLECTOR 2: WAYBACK MACHINE (1 YEAR BACKFILL) ──────────────────────────
def collect_wayback(conn, date_from, date_to):
    """
    Fetches archived weekly RSS snapshots going back 1 year.
    Each feed gets ~52 snapshots (1 per week), each snapshot
    contains ~20-50 articles = thousands of historical articles.
    Time: 30-60 mins for full backfill. Run once, then use --skip-wayback.
    """
    log("=" * 55)
    log("STEP: Wayback Machine — 1 year backfill")
    log("  Fetching ~1 snapshot per week per feed going back 52 weeks")
    log("  This runs ONCE. Use --skip-wayback on future runs.")
    log("=" * 55)

    CDX_URL = "http://web.archive.org/cdx/search/cdx"
    total   = 0

    for name, (feed_url, label) in WAYBACK_FEEDS.items():
        tag = "[REAL]" if label == 0 else "[FAKE]"
        log(f"\n  {tag} {name}")

        params = {
            "url":      feed_url,
            "output":   "json",
            "fl":       "timestamp,original",
            "from":     date_from.strftime("%Y%m%d"),
            "to":       date_to.strftime("%Y%m%d"),
            "limit":    60,
            "collapse": "timestamp:6",
        }
        try:
            r = requests.get(CDX_URL, params=params, timeout=20)
            if not r.text.strip():
                log("    No snapshots found"); continue
            snapshots = r.json()[1:]
        except Exception as e:
            log(f"    CDX error: {e}"); time.sleep(2); continue

        log(f"    {len(snapshots)} snapshots found — fetching...")
        feed_total = 0

        for i, snap in enumerate(snapshots):
            ts       = snap[0]
            archived = f"http://web.archive.org/web/{ts}/{feed_url}"
            try:
                feed = feedparser.parse(archived)
            except Exception:
                continue

            snap_new = 0
            for entry in feed.entries[:50]:
                title  = strip_html(getattr(entry, 'title', ''))
                desc   = strip_html(getattr(entry, 'summary', '') or getattr(entry, 'description', ''))
                url_   = getattr(entry, 'link', '')
                pub    = parse_date(getattr(entry, 'published', '') or getattr(entry, 'updated', ''))
                author = getattr(entry, 'author', '')
                if not title:
                    continue
                row = {
                    "id": make_id(url_, title), "collected_at": datetime.now(timezone.utc).isoformat(),
                    "source_name": name, "label": label, "title": title,
                    "description": desc[:1000], "content": None, "url": url_,
                    "published_at": pub, "author": author, "category": "archived",
                    "collection_method": "wayback", **basic_features(title, desc),
                }
                if insert(conn, row):
                    snap_new += 1

            feed_total += snap_new
            total      += snap_new
            if (i + 1) % 10 == 0:
                log(f"    [{i+1}/{len(snapshots)}] {feed_total} articles so far")
            time.sleep(DELAY)

        log(f"    Done: +{feed_total} articles from {name}")
        time.sleep(2)

    log(f"\nWayback done: {total:,} new articles\n")
    return total


# ─── COLLECTOR 3: NEWSAPI ─────────────────────────────────────────────────────
def collect_newsapi(conn, api_key, date_from, date_to):
    if not api_key:
        log("NewsAPI: no key — skipping (free key at newsapi.org)\n")
        return 0

    log("=" * 55)
    log("STEP: NewsAPI.org")
    log("=" * 55)

    effective_from = max(date_from, datetime.now(timezone.utc) - timedelta(days=29))
    from_str = effective_from.strftime("%Y-%m-%d")
    to_str   = date_to.strftime("%Y-%m-%d")
    queries  = ["politics election", "science research", "technology AI",
                "health vaccine", "economy finance", "climate environment",
                "India news", "world international"]
    total = 0

    for q in queries:
        log(f"  Query: '{q}' ...")
        try:
            r = requests.get("https://newsapi.org/v2/everything", timeout=15, params={
                "q": q, "language": "en", "from": from_str, "to": to_str,
                "sortBy": "publishedAt", "pageSize": 100, "apiKey": api_key,
            })
            data = r.json()
        except Exception as e:
            log(f"    ERROR: {e}"); continue

        if data.get("status") != "ok":
            log(f"    API error: {data.get('message')}"); continue

        count = 0
        for a in data.get("articles", []):
            title = (a.get("title") or "").replace("[Removed]", "").strip()
            if not title: continue
            desc = strip_html(a.get("description") or "")
            row  = {
                "id": make_id(a.get("url", ""), title),
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "source_name": f"newsapi_{q.split()[0]}", "label": 0,
                "title": title, "description": desc[:1000],
                "content": strip_html(a.get("content") or "")[:2000],
                "url": a.get("url", ""),
                "published_at": parse_date(a.get("publishedAt", "")),
                "author": a.get("author", ""), "category": q.split()[0],
                "collection_method": "newsapi", **basic_features(title, desc),
            }
            if insert(conn, row): count += 1

        log(f"    +{count} articles")
        total += count
        time.sleep(DELAY)

    log(f"NewsAPI done: {total:,} new articles\n")
    return total


# ─── COLLECTOR 4: NEWSDATA ────────────────────────────────────────────────────
def collect_newsdata(conn, api_key, date_from, date_to):
    if not api_key:
        log("NewsData: no key — skipping (free key at newsdata.io)\n")
        return 0

    log("=" * 55)
    log("STEP: NewsData.io archive")
    log("=" * 55)

    from_str = date_from.strftime("%Y-%m-%d")
    to_str   = date_to.strftime("%Y-%m-%d")
    topics   = ["politics", "science", "technology", "health", "world", "business", "environment"]
    total    = 0

    for topic in topics:
        log(f"  Topic: {topic} ...")
        try:
            r = requests.get("https://newsdata.io/api/1/archive", timeout=15, params={
                "apikey": api_key, "q": topic, "language": "en",
                "from_date": from_str, "to_date": to_str,
            })
            data = r.json()
        except Exception as e:
            log(f"    ERROR: {e}"); continue

        count = 0
        for a in data.get("results", []):
            title = (a.get("title") or "").strip()
            if not title: continue
            desc = strip_html(a.get("description") or "")
            row  = {
                "id": make_id(a.get("link", ""), title),
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "source_name": f"newsdata_{topic}", "label": 0,
                "title": title, "description": desc[:1000],
                "content": strip_html(a.get("content") or "")[:2000],
                "url": a.get("link", ""),
                "published_at": parse_date(a.get("pubDate", "")),
                "author": ", ".join(a.get("creator") or []),
                "category": topic, "collection_method": "newsdata",
                **basic_features(title, desc),
            }
            if insert(conn, row): count += 1

        log(f"    +{count} articles")
        total += count
        time.sleep(DELAY)

    log(f"NewsData done: {total:,} new articles\n")
    return total


# ─── STATS ───────────────────────────────────────────────────────────────────
def print_stats(conn, date_from):
    total     = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    real      = conn.execute("SELECT COUNT(*) FROM articles WHERE label=0").fetchone()[0]
    fake      = conn.execute("SELECT COUNT(*) FROM articles WHERE label=1").fetchone()[0]
    unlabeled = conn.execute("SELECT COUNT(*) FROM articles WHERE label=-1").fetchone()[0]

    def bar(v, t, w=28):
        f = int(w * v / max(t, 1))
        return "█" * f + "░" * (w - f)

    print("\n" + "=" * 55)
    print("  DATASET SUMMARY")
    print("=" * 55)
    print(f"  Window   : {date_from.date()} to today")
    print(f"  Total    : {total:,}")
    print(f"  Real (0) : {real:,}  {bar(real, total)}")
    print(f"  Fake (1) : {fake:,}  {bar(fake, total)}")
    print(f"  Unlabeled: {unlabeled:,}")

    labeled = real + fake
    if labeled > 0:
        bal = fake / labeled * 100
        print(f"\n  Balance  : {bal:.1f}% fake")
        if bal < 25:
            print("  WARNING: Very few fake articles.")
            print("  Some satire sites may be temporarily down.")
            print("  Try running again later.")
        elif 35 <= bal <= 65:
            print("  GOOD: Healthy class balance.")

    print("\n  Top sources:")
    for src, cnt in conn.execute(
        "SELECT source_name, COUNT(*) FROM articles "
        "GROUP BY source_name ORDER BY COUNT(*) DESC LIMIT 15"
    ):
        print(f"    {src:<35} {cnt:,}")

    print(f"\n  Readiness for step2:")
    if labeled >= 5000:
        print(f"  EXCELLENT — {labeled:,} labeled articles. Run step2_preprocess.py")
    elif labeled >= 2000:
        print(f"  GOOD — {labeled:,} articles. Run step2_preprocess.py")
    elif labeled >= 1000:
        print(f"  OK — {labeled:,} articles. Model will work. Run step2_preprocess.py")
    elif labeled >= 500:
        print(f"  LOW — {labeled:,} articles. Accuracy may suffer. Consider running again.")
    else:
        print(f"  TOO LOW — {labeled:,} articles. Run step1 again for more data.")
    print("=" * 55 + "\n")


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Collect news articles for fake news detection")
    parser.add_argument("--days",         type=int, default=365,
                        help="Days back to collect via RSS/Wayback (default 365)")
    parser.add_argument("--newsapi",      default="", help="NewsAPI.org key")
    parser.add_argument("--newsdata",     default="", help="NewsData.io key")
    parser.add_argument("--skip-wayback", action="store_true",
                        help="Skip Wayback Machine (faster, less historical data)")
    parser.add_argument("--rss-only",     action="store_true",
                        help="Only Kaggle + live RSS feeds (fastest option)")
    parser.add_argument("--skip-kaggle",  action="store_true",
                        help="Skip Kaggle CSV loading")
    args = parser.parse_args()

    date_to   = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=args.days)

    log("=" * 55)
    log("FAKE NEWS DATA COLLECTOR")
    log(f"Collecting past {args.days} days of news")
    log(f"Window: {date_from.date()} to {date_to.date()}")
    if args.rss_only:
        log("Mode: Kaggle + RSS only (fastest)")
    elif args.skip_wayback:
        log("Mode: Kaggle + RSS + APIs (no Wayback)")
    else:
        log("Mode: Kaggle + RSS + Wayback (full 1-year backfill)")
        log("NOTE: Wayback takes 30-60 mins. Run with --skip-wayback on future runs.")
    log("=" * 55)

    conn = init_db()

    # Load Kaggle dataset first (no internet needed once CSVs are downloaded)
    n_kaggle = 0
    if not args.skip_kaggle:
        n_kaggle = collect_kaggle(conn)
    else:
        log("Kaggle dataset: skipped (--skip-kaggle flag set)\n")

    # Live RSS feeds
    n_rss = collect_rss(conn, date_from)

    # Wayback Machine historical backfill
    n_wb = 0
    if not args.skip_wayback and not args.rss_only:
        n_wb = collect_wayback(conn, date_from, date_to)

    # Optional API sources
    n_api = 0 if args.rss_only else collect_newsapi(conn, args.newsapi, date_from, date_to)
    n_nd  = 0 if args.rss_only else collect_newsdata(conn, args.newsdata, date_from, date_to)

    log("-" * 55)
    log(f"Kaggle:{n_kaggle:,}  RSS:{n_rss:,}  Wayback:{n_wb:,}  NewsAPI:{n_api:,}  NewsData:{n_nd:,}")
    log(f"Total new this run: {n_kaggle + n_rss + n_wb + n_api + n_nd:,}")

    print_stats(conn, date_from)
    export_csv(conn)
    conn.close()
    log("Done. Run step2_preprocess.py next.")


if __name__ == "__main__":
    main()
