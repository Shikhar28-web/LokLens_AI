"""Clean all input and output records from SQLite databases."""
import sqlite3
import os

db_paths = [
    "D:/Projects/Political_lens/backend/data/loklens_ai.db",
    "D:/Projects/Political_lens/backend/data/truthlens.db"
]

tables_to_clear = [
    "evidence_graph_edges",
    "timelines",
    "verdicts",
    "image_forensics",
    "images",
    "evidence",
    "search_results",
    "claim_entities",
    "claims",
    "documents",
    "sources",
    "submissions"
]

for db_path in db_paths:
    if not os.path.exists(db_path):
        continue
    print(f"Cleaning database: {db_path}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Get actual existing tables in this DB
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = set(r[0] for r in cur.fetchall())
    
    cleared = {}
    for table in tables_to_clear:
        if table in existing_tables:
            cur.execute(f"DELETE FROM {table}")
            cleared[table] = cur.rowcount
    
    conn.commit()
    cur.execute("VACUUM")
    conn.commit()
    conn.close()
    
    print("Cleared rows:")
    for t, cnt in cleared.items():
        print(f"  {t}: {cnt} rows deleted")

print("\nDatabase cleanup complete. All input/output data is now cleared.")

