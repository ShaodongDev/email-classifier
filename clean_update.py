import sqlite3
import pandas as pd
import re

def update(file: str, db_path: str = "db/dataset.db", table: str = "emails", if_exists: str = "append") -> None:
    """
    Load a CSV with Subject, Main Category, Sub Category,
    clean the Subject text, and write it into SQLite.

    Args:
        file (str): Path to CSV file.
        db_path (str): Path to SQLite database file.
        table (str): Name of the table in SQLite.
        if_exists (str): 'append' to add rows, 'replace' to overwrite table.
    """
    df = pd.read_csv(file)  
    df = df[df["Sub Category"].notna()].copy()

    # --- Regex Clean Subject column ---
    df["Subject"] = df["Subject"].astype(str)
    df["Subject"] = df["Subject"].str.replace(r"^((fw|fwd|re):\s*)+", "", flags=re.IGNORECASE, regex=True) # Strip RE/FW/FWD prefixes
    df["Subject"] = df["Subject"].str.replace(r"\[#.*?\]\s*", "", regex=True)
    df["Subject"] = df["Subject"].str.replace(
        r"\b\d{2}[-/ ](?:[A-Z]{3}|[a-z]{3})[-/ ]\d{4}\b|\b\d{4}-\d{2}-\d{2}\b",
        "",
        regex=True
    )
    date_pattern = rf"""
    (
      \b\d{{4}}[-/]\d{{2}}[-/]\d{{2}}\b                         
    | \b\d{{1,2}}[-/]\d{{1,2}}[-/]\d{{2,4}}\b                     
    )
    """
    df["Subject"] = df["Subject"].str.replace(date_pattern, "", flags=re.IGNORECASE | re.VERBOSE, regex=True) # remove data YYYY-MM-DD or YYYY/MM/DD or  DD/MM/YYYY or MM/DD/YYYY
    
    df["Subject"] = df["Subject"].str.replace(r"(?:\bID\b[: ]*)?#\d+\b|\bID\b[: ]*\d+\b", "", flags=re.IGNORECASE, regex=True) # Remove naked ticket/ID numbers
    df["Subject"] = df["Subject"].str.replace(r"\b[A-Z]{2}\d{2}\b", "", regex=True)    # remove region code e.g., JP10, UK10
    df["Subject"] = df["Subject"].str.replace(r"\s*[-–—|:]+\s*$", "", regex=True)      # trailing delimiters
    df["Subject"] = df["Subject"].str.replace(r"\s*[-–—|:]+\s*", " - ", regex=True)    # normalize internal separators
    df["Subject"] = df["Subject"].str.replace(r"\s+", " ", regex=True).str.strip()

    # Keep only required columns
    df = df[["Subject", "Main Category", "Sub Category"]]
    rows, _ = df.shape

    # --- Write to SQLite ---
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    df.to_sql(table, conn, if_exists=if_exists, index=False)

    cur.execute(f'SELECT COUNT(*) FROM "{table}";')
    row_DB = cur.fetchone()[0]
    conn.close()

    print(f"✅ {db_path} has been updated with {rows} new entries ({if_exists} mode).")
    print(f"✅ There are now {row_DB} total rows in table '{table}'.")
    print("\nHere’s a preview of the table:")
    print(df.head()["Subject"])