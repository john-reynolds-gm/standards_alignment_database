"""
build_database.py

Loads the three Learnosity/CCSS source files into a normalized SQLite
database following the states / ccss_standards / state_standards / crosswalk
schema.

Inputs (set paths below):
    - learnosity_combined.csv   (output of learnosity_pipeline.py)
    - ccss_m_canon.csv          (canonical CCSS standards, with text)
    - all_states_just_standards.csv  (state standard text reference)

Output:
    - learnosity.db  (SQLite database)

Requirements:
    pip install pandas
"""

import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────

COMBINED_CSV = "learnosity_combined.csv"
CANON_CSV = "ccss_m_canon.csv"
STATE_TEXT_CSV = "all_states_just_standards.csv"
DB_PATH = "learnosity.db"

# ─────────────────────────────────────────────────────────────────────────────


def strip_hs_prefix(code: str) -> str:
    """Remove a leading 'HS' from a CCSS code, e.g. 'HSA.APR.A.1' -> 'A.APR.A.1'."""
    if pd.isna(code):
        return code
    return re.sub(r'^HS', '', code)


# A grade token is the anchor where a state's own standard id begins: a single
# grade (K, 1-8), a grade band (9-12, 9-10, 11-12), or the 'HS' band marker.
# Restricted to 1-2 digit grades so 4-digit year segments (e.g. 'IN.2023...')
# are not mistaken for grades.
GRADE_TOKEN = re.compile(r'^(K|\d{1,2}|\d{1,2}-\d{1,2}|HS)$')


def normalize_separators(code: str) -> str:
    """Canonicalize the domain-cluster separator: dashes -> dots (A-APR -> A.APR)."""
    return code.replace('-', '.') if isinstance(code, str) else code


def isolate_standard_id(new_tag_value: str) -> str:
    """
    Strip the leading state/framework/subject metadata from a Learnosity
    newTagValue, returning the state-local standard id starting at the first
    grade token. 'SC.CCRS.MA.6.GM.3' -> '6.GM.3'; 'OH.Math.1.G' -> '1.G'.
    Falls back to the whole value when no grade token is present (HS-only codes).
    """
    if not isinstance(new_tag_value, str):
        return new_tag_value
    parts = new_tag_value.split('.')
    for i, p in enumerate(parts):
        if GRADE_TOKEN.match(p):
            return '.'.join(parts[i:])
    return new_tag_value  # no grade token (e.g. 'MA.AR.A-APR') — best-effort


def load_canon(canon_path: Path) -> pd.DataFrame:
    """
    Load the CCSS canon, strip the HS prefix from id/parent_ccss_id, and
    resolve the self-referencing 'no parent' sentinel (parent == id) to a
    true null. Assigns a sequential ccss_id and resolves parent_id via a
    self-join on the cleaned code.
    """
    canon = pd.read_csv(canon_path, dtype=str)

    canon["code"] = canon["id"].apply(strip_hs_prefix)
    canon["parent_code"] = canon["parent_ccss_id"].apply(strip_hs_prefix)

    # Sentinel: parent_code == code means "no real parent" (no Domain/Cluster
    # rows exist yet in canon) -> convert to NaN
    canon.loc[canon["parent_code"] == canon["code"], "parent_code"] = pd.NA

    canon = canon.reset_index(drop=True)
    canon["ccss_id"] = canon.index + 1  # sequential PK, 1-indexed

    code_to_id = dict(zip(canon["code"], canon["ccss_id"]))
    canon["parent_id"] = canon["parent_code"].map(code_to_id)

    canon["is_subpart"] = canon["is_subpart"].str.upper() == "TRUE"

    ccss_standards = canon[
        ["ccss_id", "code", "grade", "domain", "domain_name", "cluster",
         "standard", "is_subpart", "text", "parent_id"]
    ].copy()

    return ccss_standards


def build_states(combined: pd.DataFrame, state_text: pd.DataFrame) -> pd.DataFrame:
    """
    Build the states lookup table from the union of states appearing in the
    combined crosswalk data and the state standards text file. standards_year
    is taken from the combined data where available; states with no year
    annotation (the majority) get a null year.
    """
    crosswalk_states = combined[["state", "standards_year"]].drop_duplicates()
    crosswalk_states["standards_year"] = pd.to_numeric(
        crosswalk_states["standards_year"], errors="coerce"
    )

    # States that only appear in the text reference file (no crosswalk yet)
    text_only_states = set(state_text["state"].unique()) - set(combined["state"].unique())
    text_only_df = pd.DataFrame({
        "state": sorted(text_only_states),
        "standards_year": pd.NA,
    })

    states = pd.concat([crosswalk_states, text_only_df], ignore_index=True)
    states["standards_year"] = pd.to_numeric(states["standards_year"], errors="coerce")
    # year_key: sentinel (0) substituted for missing year so it can be used
    # as a reliable join/lookup key — NaN famously never equals NaN, which
    # would silently break every lookup for states with no year annotation.
    states["year_key"] = states["standards_year"].fillna(0)
    states = states.drop_duplicates(subset=["state", "year_key"]).reset_index(drop=True)
    states["state_id"] = states.index + 1

    return states[["state_id", "state", "standards_year", "year_key"]].rename(
        columns={"state": "abbreviation"}
    )


def build_state_standards(
    combined: pd.DataFrame, state_text: pd.DataFrame, states: pd.DataFrame
) -> pd.DataFrame:
    """
    Build one row per unique (state, standards_year, state standard code)
    combination found in the combined crosswalk data, parsing the agency
    code and level name out of newTagName, and left-joining in standard_text
    and grade from the state text reference file where available.
    """
    parsed = combined["newTagName"].str.extract(r'^(\S+)\s+(.+)$')
    parsed.columns = ["agency_code", "level_name"]

    work = pd.concat(
        [combined[["state", "standards_year", "newTagValue"]], parsed], axis=1
    ).rename(columns={"newTagValue": "standard_code"})

    work["standards_year_num"] = pd.to_numeric(work["standards_year"], errors="coerce")
    work["year_key"] = work["standards_year_num"].fillna(0)

    # Isolate the state-local standard id from the prefixed newTagValue, then
    # canonicalize separators so it can be matched against the reference. The raw
    # newTagValue is retained as standard_code for provenance and crosswalk keying.
    work["standard_code_id"] = (
        work["standard_code"].apply(isolate_standard_id).apply(normalize_separators)
    )

    state_standards = work.drop_duplicates(
        subset=["state", "year_key", "standard_code"]
    ).reset_index(drop=True)

    # Join state_id (matching on the sentinel-filled year_key, not the raw
    # nullable year, to avoid the NaN-never-equals-NaN trap)
    states_for_merge = states.rename(columns={"abbreviation": "state"})[
        ["state_id", "state", "year_key"]
    ]
    state_standards = state_standards.merge(
        states_for_merge,
        on=["state", "year_key"],
        how="left",
    )

    # Join standard text/grade from the reference file (no year dimension there,
    # so this joins on state + code only). The reference standard_id is separator-
    # normalized to match the isolated standard_code_id.
    ref = state_text.copy()
    ref["standard_code_id"] = ref["standard_id"].apply(normalize_separators)
    state_standards = state_standards.merge(
        ref[["state", "standard_code_id", "standard_text", "grade"]].drop_duplicates(
            subset=["state", "standard_code_id"]
        ),
        on=["state", "standard_code_id"],
        how="left",
    )

    state_standards = state_standards.reset_index(drop=True)
    state_standards["state_standard_id"] = state_standards.index + 1

    return state_standards[
        ["state_standard_id", "state_id", "state", "year_key", "agency_code",
         "level_name", "standard_code", "standard_code_id", "grade", "standard_text"]
    ]


def build_crosswalk(
    combined: pd.DataFrame,
    ccss_standards: pd.DataFrame,
    state_standards_full: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Build the crosswalk junction table by parsing the CCSS code out of
    targetTags (stripping the HS prefix) and resolving both sides to their
    surrogate keys. Rows whose CCSS code isn't in canon yet (Domain/Cluster
    level entries) are skipped and counted for the report.
    """
    extracted = combined["targetTags"].str.extract(
        r'^CCSS Math (Domain|Cluster|Standard|Child Standard):\s*([^/]+)'
    )
    extracted.columns = ["ccss_level", "ccss_code_raw"]

    work = pd.concat([combined, extracted], axis=1)
    work["ccss_code"] = work["ccss_code_raw"].str.strip().apply(strip_hs_prefix)
    work["standards_year_num"] = pd.to_numeric(work["standards_year"], errors="coerce")
    work["year_key"] = work["standards_year_num"].fillna(0)

    code_to_ccss_id = dict(zip(ccss_standards["code"], ccss_standards["ccss_id"]))
    work["ccss_id"] = work["ccss_code"].map(code_to_ccss_id)

    # Resolve state_standard_id via (state, year_key, standard_code)
    key_to_ssid = {}
    for _, row in state_standards_full.iterrows():
        key_to_ssid[(row["state"], row["year_key"], row["standard_code"])] = row["state_standard_id"]
    work["state_standard_id"] = work.apply(
        lambda r: key_to_ssid.get((r["state"], r["year_key"], r["newTagValue"])),
        axis=1,
    )

    unmatched_ccss = work[work["ccss_id"].isna()]
    report = {
        "total_rows": len(work),
        "unmatched_ccss_count": len(unmatched_ccss),
        "unmatched_ccss_by_level": unmatched_ccss["ccss_level"].value_counts().to_dict(),
        "unmatched_state_standard_count": work["state_standard_id"].isna().sum(),
    }

    matched = work.dropna(subset=["ccss_id", "state_standard_id"]).reset_index(drop=True)
    matched["crosswalk_id"] = matched.index + 1
    matched["ccss_id"] = matched["ccss_id"].astype(int)
    matched["state_standard_id"] = matched["state_standard_id"].astype(int)

    crosswalk = matched[
        ["crosswalk_id", "ccss_id", "state_standard_id", "source_file"]
    ]

    return crosswalk, report


def write_database(
    db_path: Path,
    states: pd.DataFrame,
    ccss_standards: pd.DataFrame,
    state_standards: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> None:
    conn = sqlite3.connect(db_path)

    states.to_sql("states", conn, if_exists="replace", index=False)
    ccss_standards.to_sql("ccss_standards", conn, if_exists="replace", index=False)
    state_standards.to_sql("state_standards", conn, if_exists="replace", index=False)
    crosswalk.to_sql("crosswalk", conn, if_exists="replace", index=False)

    cur = conn.cursor()
    cur.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_ccss_parent ON ccss_standards(parent_id);
        CREATE INDEX IF NOT EXISTS idx_state_standards_state ON state_standards(state_id);
        CREATE INDEX IF NOT EXISTS idx_crosswalk_ccss ON crosswalk(ccss_id);
        CREATE INDEX IF NOT EXISTS idx_crosswalk_state_standard ON crosswalk(state_standard_id);
        """
    )
    conn.commit()
    conn.close()


def main():
    combined_path = Path(COMBINED_CSV)
    canon_path = Path(CANON_CSV)
    state_text_path = Path(STATE_TEXT_CSV)

    for p in (combined_path, canon_path, state_text_path):
        if not p.exists():
            print(f"ERROR: Required input not found — {p}")
            sys.exit(1)

    print("Loading source files...")
    combined = pd.read_csv(combined_path, dtype=str)
    state_text = pd.read_csv(state_text_path, dtype=str)

    print("Building ccss_standards from canon...")
    ccss_standards = load_canon(canon_path)

    print("Building states...")
    states = build_states(combined, state_text)

    print("Building state_standards...")
    state_standards_full = build_state_standards(combined, state_text, states)

    print("Building crosswalk...")
    crosswalk, report = build_crosswalk(combined, ccss_standards, state_standards_full)

    state_standards = state_standards_full[
        ["state_standard_id", "state_id", "agency_code", "level_name",
         "standard_code", "standard_code_id", "grade", "standard_text"]
    ]

    print("Writing SQLite database...")
    states_out = states.drop(columns=["year_key"])
    write_database(Path(DB_PATH), states_out, ccss_standards, state_standards, crosswalk)

    print(f"\n{'=' * 64}")
    print("  BUILD SUMMARY")
    print(f"{'=' * 64}")
    print(f"  states            : {len(states):,} rows")
    print(f"  ccss_standards    : {len(ccss_standards):,} rows")
    print(f"  state_standards   : {len(state_standards):,} rows")
    print(f"  crosswalk         : {len(crosswalk):,} rows")
    print(f"\n  Crosswalk source rows scanned : {report['total_rows']:,}")
    print(f"  Skipped — CCSS code not in canon yet : {report['unmatched_ccss_count']:,}")
    for level, count in report["unmatched_ccss_by_level"].items():
        print(f"      {level}: {count:,}")
    print(f"  Skipped — state standard not resolved : {report['unmatched_state_standard_count']:,}")

    # standard_text coverage per state: surfaces framework/vintage mismatches
    # (e.g. MD/ME/PA/SC) that leave standard_text unpopulated despite clean id
    # isolation, rather than letting them fail silently.
    cov = state_standards_full.copy()
    cov["has_text"] = cov["standard_text"].notna()
    total_text = int(cov["has_text"].sum())
    print(f"\n{'=' * 64}")
    print(f"  STANDARD_TEXT COVERAGE  —  {total_text:,}/{len(cov):,} "
          f"({total_text / len(cov):.0%}) state_standards rows matched to reference text")
    print(f"{'=' * 64}")
    states_with_ref = set(state_text.dropna(subset=["standard_id"])["state"].unique())
    per_state = cov.groupby("state")["has_text"].agg(["sum", "count"])
    per_state["pct"] = per_state["sum"] / per_state["count"]
    for state, row in per_state.sort_values("pct").iterrows():
        if state not in states_with_ref:
            flag = "  ·  no reference standards for this state"
        elif row["pct"] < 0.20:
            flag = "  ⚠️  framework/vintage mismatch?"
        else:
            flag = ""
        print(f"    {state:<4} {int(row['sum']):>4}/{int(row['count']):<4}  "
              f"({row['pct']:>4.0%}){flag}")

    print(f"\n  Database written to: {DB_PATH}")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
