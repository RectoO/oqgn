import time
import os
import sqlite3

from src.constants import DB_PATH


def init_db():
    if not os.path.isfile(DB_PATH):
        raise AssertionError("Missing counter db")

    print("Initializing counter database", flush=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=FULL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            delta INTEGER NOT NULL,
            cumulative INTEGER NOT NULL
        );
    """
    )
    conn.commit()
    conn.close()


def increment(delta):
    ts = int(time.time())
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("BEGIN IMMEDIATE;")  # grab write lock
    cur = conn.execute("SELECT cumulative FROM ledger ORDER BY id DESC LIMIT 1;")
    row = cur.fetchone()
    last_total = row[0] if row else 0
    total = last_total + delta
    conn.execute(
        "INSERT INTO ledger (ts, delta, cumulative) VALUES (?, ?, ?)",
        (ts, delta, total),
    )
    conn.commit()
    conn.close()
    return total


def get_total():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cur = conn.execute("SELECT cumulative FROM ledger ORDER BY id DESC LIMIT 1;")
    row = cur.fetchone()
    conn.close()
    return row[0] if row else 0
