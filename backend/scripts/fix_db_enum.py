"""
Fix script: Migrate any invalid source_type values ('web', 'fact_checker', etc.)
in the loklens_ai.db SQLite database to valid enum values.
"""
import sqlite3

VALID_TYPES = {
    "government", "official_org", "news", "academic",
    "research", "specialist", "blog", "social_media", "unknown"
}

TYPE_REMAP = {
    "web": "unknown",
    "fact_checker": "official_org",
    "website": "unknown",
    "general": "unknown",
    "other": "unknown",
}

db_path = "D:/Projects/Political_lens/backend/data/loklens_ai.db"
print(f"Processing: {db_path}")

conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Show all tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", [r[0] for r in cur.fetchall()])

# Check current state
cur.execute("SELECT source_type, COUNT(*) FROM sources GROUP BY source_type")
rows = cur.fetchall()
print("\nCurrent source_type distribution:")
for row in rows:
    marker = " <-- INVALID" if row[0] not in VALID_TYPES else " OK"
    print(f"  {repr(row[0])}: {row[1]}{marker}")

# Fix all known bad mappings
fixed = 0
for bad_val, good_val in TYPE_REMAP.items():
    cur.execute(
        "UPDATE sources SET source_type = ? WHERE source_type = ?",
        (good_val, bad_val)
    )
    count = cur.rowcount
    if count:
        print(f"  Fixed {count} rows: {repr(bad_val)} -> {repr(good_val)}")
        fixed += count

# Fix NULL / empty values
cur.execute("UPDATE sources SET source_type = 'unknown' WHERE source_type IS NULL OR source_type = ''")
count = cur.rowcount
if count:
    print(f"  Fixed {count} NULL/empty rows -> 'unknown'")
    fixed += count

# Fix any other remaining invalid values
cur.execute("SELECT DISTINCT source_type FROM sources")
remaining = [r[0] for r in cur.fetchall()]
for val in remaining:
    if val not in VALID_TYPES:
        cur.execute(
            "UPDATE sources SET source_type = 'unknown' WHERE source_type = ?",
            (val,)
        )
        count = cur.rowcount
        if count:
            print(f"  Fixed {count} rows: {repr(val)} -> 'unknown'")
            fixed += count

conn.commit()
print(f"\nTotal rows fixed: {fixed}")

# Verify after fix
cur.execute("SELECT source_type, COUNT(*) FROM sources GROUP BY source_type")
rows = cur.fetchall()
print("\nAfter fix:")
for row in rows:
    marker = " OK" if row[0] in VALID_TYPES else " STILL INVALID!"
    print(f"  {repr(row[0])}: {row[1]}{marker}")

# Also check submissions table for 'error' status
cur.execute("SELECT status, COUNT(*) FROM submissions GROUP BY status")
rows = cur.fetchall()
print("\nSubmission statuses:")
for row in rows:
    print(f"  {repr(row[0])}: {row[1]}")

conn.close()
print("\nDone.")
