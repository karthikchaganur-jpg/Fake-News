import csv
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# ─── PATHS ───────────────────────────────────────────────────────────────────
DATA_DIR    = Path("news_dataset")
INPUT_CSV   = DATA_DIR / "preprocessed.csv"
MODEL_DIR   = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_PATH = DATA_DIR / "evaluation_report.txt"
LOG_CSV     = DATA_DIR / "training_log.csv"

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def divider(char="─", width=55):
    print(char * width)

def log_to_file(text, path=REPORT_PATH):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def print_and_log(text):
    print(text)
    log_to_file(text)

# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    # Clear old report
    REPORT_PATH.write_text("")

    header = f"""
{'='*55}
  FAKE NEWS DETECTOR — MODEL TRAINING REPORT
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*55}
"""
    print_and_log(header)

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    if not INPUT_CSV.exists():
        print("ERROR: news_dataset/preprocessed.csv not found.")
        print("Run step2_preprocess.py first.")
        return

    df = pd.read_csv(INPUT_CSV)
    print_and_log(f"Loaded {len(df):,} articles from {INPUT_CSV}")

    if len(df) < 100:
        print("WARNING: Very few articles. Collect more data for reliable results.")

    X = df["cleaned_text"].fillna("").astype(str)
    y = df["label"].astype(int)

    real = (y == 0).sum()
    fake = (y == 1).sum()
    print_and_log(f"Class distribution — Real: {real:,}  Fake: {fake:,}")
    print_and_log(f"Class balance      — {fake/(real+fake)*100:.1f}% fake\n")

    # ── 2. Train/Test Split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print_and_log(f"Train size: {len(X_train):,}   Test size: {len(X_test):,}")
    print_and_log(f"(80% train / 20% test, stratified by label)\n")

    # ── 3. TF-IDF Vectorizer ─────────────────────────────────────────────────
    print_and_log("TF-IDF Configuration:")
    print_and_log("  max_features = 10,000")
    print_and_log("  ngram_range  = (1, 2)  [unigrams + bigrams]")
    print_and_log("  sublinear_tf = True    [log normalization]\n")

    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)

    # Save vectorizer
    joblib.dump(tfidf, MODEL_DIR / "tfidf_vectorizer.pkl")
    print_and_log(f"Vocabulary size: {len(tfidf.vocabulary_):,} features")
    print_and_log(f"Vectorizer saved → {MODEL_DIR}/tfidf_vectorizer.pkl\n")

    # ── 4. Define 3 Models (as per PPT) ──────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=42
        ),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "SVM (Linear)": CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, random_state=42)
        ),
    }

    results   = {}
    log_rows  = []
    best_f1   = 0
    best_name = ""
    best_pipe = None

    # ── 5. Train & Evaluate Each Model ───────────────────────────────────────
    for model_name, clf in models.items():
        divider()
        print_and_log(f"  MODEL: {model_name}")
        divider()

        # Train
        clf.fit(X_train_tfidf, y_train)

        # Predict
        y_pred     = clf.predict(X_test_tfidf)
        y_pred_proba = clf.predict_proba(X_test_tfidf)[:, 1] if hasattr(clf, 'predict_proba') else None

        # Core metrics
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)

        results[model_name] = {
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
        }

        # 5-fold cross validation on training set
        cv_scores = cross_val_score(clf, X_train_tfidf, y_train, cv=5, scoring="f1")

        print_and_log(f"\n  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
        print_and_log(f"  Precision : {prec:.4f}")
        print_and_log(f"  Recall    : {rec:.4f}")
        print_and_log(f"  F1-Score  : {f1:.4f}")
        print_and_log(f"\n  5-Fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        print_and_log(f"\n  Confusion Matrix:")
        print_and_log(f"  {'':12} Predicted Real  Predicted Fake")
        print_and_log(f"  Actual Real  {cm[0][0]:>12,}   {cm[0][1]:>12,}")
        print_and_log(f"  Actual Fake  {cm[1][0]:>12,}   {cm[1][1]:>12,}")

        tn, fp, fn, tp = cm.ravel()
        print_and_log(f"\n  True Positives  (TP): {tp:,}  — fake correctly caught")
        print_and_log(f"  True Negatives  (TN): {tn:,}  — real correctly passed")
        print_and_log(f"  False Positives (FP): {fp:,}  — real wrongly flagged")
        print_and_log(f"  False Negatives (FN): {fn:,}  — fake missed")

        print_and_log(f"\n  Full Classification Report:")
        report = classification_report(y_test, y_pred,
                                       target_names=["Real (0)", "Fake (1)"])
        print_and_log(report)

        # Save model
        safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        model_path = MODEL_DIR / f"model_{safe_name}.pkl"
        joblib.dump(clf, model_path)
        print_and_log(f"  Model saved → {model_path}")

        log_rows.append({
            "model": model_name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "cv_f1_mean": round(cv_scores.mean(), 4),
            "cv_f1_std": round(cv_scores.std(), 4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        })

        if f1 > best_f1:
            best_f1   = f1
            best_name = model_name
            best_pipe = clf

    # ── 6. Comparison Table ───────────────────────────────────────────────────
    divider("=")
    print_and_log("  MODEL COMPARISON TABLE")
    divider("=")
    header_row = f"  {'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    print_and_log(header_row)
    divider()
    for name, m in results.items():
        marker = " ← BEST" if name == best_name else ""
        row = (f"  {name:<25} {m['accuracy']:>9.4f} {m['precision']:>10.4f}"
               f" {m['recall']:>8.4f} {m['f1']:>8.4f}{marker}")
        print_and_log(row)
    divider("=")

    # ── 7. Save Best Model as Primary ────────────────────────────────────────
    best_path = DATA_DIR / "best_model.pkl"
    joblib.dump(best_pipe, best_path)
    joblib.dump(tfidf, DATA_DIR / "tfidf_vectorizer.pkl")
    print_and_log(f"\n  Best model : {best_name}  (F1={best_f1:.4f})")
    print_and_log(f"  Saved as   : {best_path}")
    print_and_log(f"  Vectorizer : {DATA_DIR}/tfidf_vectorizer.pkl")

    # ── 8. Top TF-IDF Features (Logistic Regression) ─────────────────────────
    if "Logistic Regression" in models:
        lr_model = models["Logistic Regression"]
        feature_names = tfidf.get_feature_names_out()
        coefs = lr_model.coef_[0]
        top_n = 20

        top_fake_idx = np.argsort(coefs)[-top_n:][::-1]
        top_real_idx = np.argsort(coefs)[:top_n]

        print_and_log(f"\n  TOP {top_n} FEATURES → FAKE NEWS (high LR weight)")
        divider()
        for idx in top_fake_idx:
            print_and_log(f"  {feature_names[idx]:<30} weight={coefs[idx]:+.4f}")

        print_and_log(f"\n  TOP {top_n} FEATURES → REAL NEWS (low LR weight)")
        divider()
        for idx in top_real_idx:
            print_and_log(f"  {feature_names[idx]:<30} weight={coefs[idx]:+.4f}")

    # ── 9. Save training log CSV ──────────────────────────────────────────────
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        w.writeheader()
        w.writerows(log_rows)
    print_and_log(f"\n  Training log saved → {LOG_CSV}")
    print_and_log(f"  Full report saved  → {REPORT_PATH}")
    divider("=")
    print_and_log("\nRun step4_predict.py to classify new articles.")

if __name__ == "__main__":
    main()
