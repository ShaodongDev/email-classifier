import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from datetime import datetime
import joblib
from typing import Optional

def train_and_evaluate(
    db_path: str = "dataset.db",
    table: str = "emails",
    report_file: Optional[str] = "classification_report.txt",
    test_size: float = 0.15,
    random_state: Optional[int] = None,
):
    """
    Load data from SQLite, train TF-IDF + MultiOutputRandomForest,
    evaluate on a holdout split, optionally save artifacts.

    Returns dict with:
      - 'vectorizer': fitted TfidfVectorizer
      - 'model'     : fitted MultiOutputClassifier
      - 'report_main': str
      - 'report_sub' : str
      - 'X_train_tfidf', 'X_test_tfidf' (sparse matrices)
      - 'y_train', 'y_test' (DataFrames)
    """
    # --- Load data ---
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT "Subject", "Main Category", "Sub Category"
    FROM "{table}"
    WHERE "Sub Category" IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # --- Train/Test Split ---
    X = df["Subject"]
    y = df[["Main Category", "Sub Category"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # --- Vectorize ---
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # --- Train multi-output classifier (one model per target) ---
    model = MultiOutputClassifier(RandomForestClassifier(random_state=random_state))
    model.fit(X_train_tfidf, y_train)

    # --- Evaluate ---
    y_pred = model.predict(X_test_tfidf)
    report_main = classification_report(y_test["Main Category"], y_pred[:, 0], zero_division=0)
    report_sub  = classification_report(y_test["Sub Category"],  y_pred[:, 1], zero_division=0)
    
    print("Main Category Report:\n", report_main)
    print("Sub Category Report:\n", report_sub)

    # --- Optionally save reports ---
    if report_file:
        now = datetime.now().isoformat(timespec="seconds")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(now + "\n")
            f.write("Main Category report:\n")
            f.write(report_main + "\n\n")
            f.write("Sub Category report:\n")
            f.write(report_sub)

    # --- Save the model ---
    joblib.dump(vectorizer, "model/tfidf_vectorizer.joblib")
    joblib.dump(model, "model/multioutput_classifier.joblib")
