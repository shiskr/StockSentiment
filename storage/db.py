import sqlite3
from datetime import datetime

DB_PATH = "sentiment.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sentiment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            buy INTEGER,
            sell INTEGER,
            hold INTEGER,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_result(ticker, buy, sell, hold):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO sentiment (ticker, buy, sell, hold, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (ticker, buy, sell, hold, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()