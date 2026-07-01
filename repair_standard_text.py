"""
repair_standard_text.py

Populates the standard_text and grade columns in the state_standards table
from all_states_just_standards.csv, which the original build_database.py
failed to load correctly.

Run from the standards_alignment_database folder:
    python3 repair_standard_text.py
"""

import sqlite3
import pandas as pd

DB_PATH  = "learnosity.db"
REF_PATH = "all_states_just_standards.csv"


# ── LOAD REFERENCE CSV ────────────────────────────────────────────────────────

ref = pd.read_csv(REF_PATH)
# Expected columns: state, grade, standard_id, standard_text

print(f"Reference CSV: {len(ref):,} rows across {ref['state'].nunique()} states.")


# ── MATCH TO DATABASE ROWS ────────────────────────────────────────────────────

with sqlite3.connect(DB_PATH) as conn:

    # Pull just the columns we need for matching — no need to load full rows.
    ss = pd.read_sql(
        """
        SELECT ss.state_standard_id, s.abbreviation, ss.standard_code
        FROM   state_standards ss
        JOIN   states s ON ss.state_id = s.state_id
        """,
        conn,
    )

    # Inner join on state abbreviation + standard code.
    # Rows in state_standards with no match in the CSV stay untouched (NULL).
    merged = ss.merge(
        ref[["state", "grade", "standard_id", "standard_text"]],
        left_on=["abbreviation", "standard_code"],
        right_on=["state", "standard_id"],
        how="inner",
    )

    print(
        f"Matched {len(merged):,} rows across "
        f"{merged['abbreviation'].nunique()} states."
    )
    if merged.empty:
        print("Nothing to update — check that standard codes match between the CSV and DB.")
        raise SystemExit(1)

    # Build the update list: (standard_text, grade, state_standard_id)
    updates = list(zip(
        merged["standard_text"].tolist(),
        merged["grade"].tolist(),
        merged["state_standard_id"].tolist(),
    ))

    # ── APPLY UPDATES ─────────────────────────────────────────────────────────

    cursor = conn.cursor()
    cursor.executemany(
        "UPDATE state_standards SET standard_text = ?, grade = ? WHERE state_standard_id = ?",
        updates,
    )
    conn.commit()
    print(f"Updated {len(updates):,} rows in state_standards.")

    # ── VERIFY ────────────────────────────────────────────────────────────────

    result = pd.read_sql(
        """
        SELECT
            s.abbreviation,
            COUNT(*)                                                          AS total,
            SUM(CASE WHEN ss.standard_text IS NOT NULL THEN 1 ELSE 0 END)    AS has_text,
            SUM(CASE WHEN ss.grade         IS NOT NULL THEN 1 ELSE 0 END)    AS has_grade
        FROM state_standards ss
        JOIN states s ON ss.state_id = s.state_id
        GROUP BY s.abbreviation
        ORDER BY s.abbreviation
        """,
        conn,
    )
    print("\nVerification:")
    print(result.to_string())
