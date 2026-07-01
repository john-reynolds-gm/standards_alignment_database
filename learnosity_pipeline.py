"""
learnosity_pipeline.py

Discovers all CSVs inside Learnosity tagging folders under a synced SharePoint
root, audits their column structures, and (on confirmation) combines the
consistent ones into a single output CSV.

Requirements:
    pip install pandas
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


# ── CONFIG ────────────────────────────────────────────────────────────────────

ROOT_DIR = r"/Users/john.reynolds/Library/CloudStorage/OneDrive-SharedLibraries-GreatMindsPBC/Eureka Math 2 - State Standards Data"

OUTPUT_CSV = "learnosity_combined.csv"

# All known folder-name suffixes that contain Learnosity tagging CSVs.
# Add new variants here if the audit report flags unknown Learnosity folders.
LEARNOSITY_FOLDER_SUFFIXES = (
    "Learnosity Bulk Tagging CSVs",
    "Learnosity Tagging Sheets",
    "Learnosity Bulk Tagging for EM2 LA",
    "Learnosity Tagging sheets",
    "Learnosity Bulk Taggings CSVs"
)

# Files whose full path contains any of these strings are silently skipped.
# Use this to exclude subfolders that are known experiments or duplicates.
EXCLUDE_PATH_FRAGMENTS = [
    "Assessment Anchors",   # PA experiment — not used for other states
]

# ─────────────────────────────────────────────────────────────────────────────


def find_learnosity_csvs(root: Path) -> tuple[list[Path], list[Path]]:
    """
    Walk the folder tree and return:
      - csv_files   : CSVs inside any folder matching a known suffix, minus exclusions
      - excluded    : CSVs that were found but skipped due to EXCLUDE_PATH_FRAGMENTS
    """
    csv_files: list[Path] = []
    excluded:  list[Path] = []

    for folder in root.rglob("*"):
        if not folder.is_dir():
            continue
        if not any(folder.name.endswith(suffix) for suffix in LEARNOSITY_FOLDER_SUFFIXES):
            continue
        for csv_file in folder.rglob("*.csv"):
            if any(frag in str(csv_file) for frag in EXCLUDE_PATH_FRAGMENTS):
                excluded.append(csv_file)
            else:
                csv_files.append(csv_file)

    return sorted(csv_files), sorted(excluded)


def find_unknown_learnosity_folders(root: Path) -> list[Path]:
    """
    Find folders whose name contains 'Learnosity' but don't match any known
    suffix. These are potential missed sources and surface in the audit report.
    """
    unknown = []
    for folder in root.rglob("*"):
        if not folder.is_dir():
            continue
        if "Learnosity" not in folder.name:
            continue
        if any(folder.name.endswith(suffix) for suffix in LEARNOSITY_FOLDER_SUFFIXES):
            continue
        unknown.append(folder)
    return sorted(unknown)


def read_headers(csv_file: Path) -> tuple[str, ...] | None:
    """
    Return the header row as a cleaned tuple. Tries UTF-8 with BOM first
    (common for Excel exports), then falls back to latin-1. Returns None if
    the file can't be read or has no headers.
    """
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with open(csv_file, encoding=encoding, newline="") as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                if headers:
                    return tuple(h.strip() for h in headers)
        except Exception:
            continue
    return None


def audit(csv_files: list[Path]) -> dict:
    """
    Group files by their column signature.
    Returns:
        structure_groups  — {columns_tuple: [Path, ...]}
        unreadable        — [Path, ...]
    """
    structure_groups: dict[tuple, list[Path]] = defaultdict(list)
    unreadable: list[Path] = []

    for f in csv_files:
        headers = read_headers(f)
        if headers is None:
            unreadable.append(f)
        else:
            structure_groups[headers].append(f)

    return {
        "structure_groups": dict(structure_groups),
        "unreadable": unreadable,
    }


def infer_state(csv_file: Path) -> str:
    """
    Extract a clean two-letter state abbreviation from the Learnosity folder
    name in the file's path. Strips year annotations in all known formats:
        "AR (2016r2019)"  →  "AR"
        "ND (2017)"       →  "ND"
        "WV_2015_"        →  "WV"
        "IN 2023"         →  "IN"
    """
    for part in csv_file.parts:
        for suffix in LEARNOSITY_FOLDER_SUFFIXES:
            if part.endswith(suffix):
                raw = part[: -len(suffix)].strip()
                # Strip (YYYY...) and (YYYYrYYYY) patterns
                raw = re.sub(r'\s*\([^)]*\d{4}[^)]*\)', '', raw)
                # Strip _YYYY_ and _YYYY patterns
                raw = re.sub(r'_\d{4}_?', '', raw)
                # Strip trailing bare year (e.g. "IN 2023")
                raw = re.sub(r'\s+\d{4}$', '', raw)
                return raw.strip()
    return "Unknown"


def infer_standards_year(csv_file: Path) -> str:
    """
    Extract the first four-digit year from the Learnosity folder name, if one
    exists. Returns an empty string when there is no year annotation.
        "AR (2016r2019)"  →  "2016"
        "ND (2017)"       →  "2017"
        "WV_2015_"        →  "2015"
        "IN 2023"         →  "2023"
        "AK"              →  ""
    """
    for part in csv_file.parts:
        for suffix in LEARNOSITY_FOLDER_SUFFIXES:
            if part.endswith(suffix):
                raw = part[: -len(suffix)]
                match = re.search(r'\d{4}', raw)
                return match.group(0) if match else ""
    return ""


def print_audit_report(
    audit_result: dict,
    excluded: list[Path],
    unknown_folders: list[Path],
) -> None:
    groups     = audit_result["structure_groups"]
    unreadable = audit_result["unreadable"]
    total      = sum(len(v) for v in groups.values()) + len(unreadable)

    print(f"\n{'=' * 64}")
    print(f"  AUDIT REPORT  —  {total} CSV(s) scanned")
    print(f"{'=' * 64}")

    if not groups:
        print("No readable CSVs found.")
    else:
        sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
        majority_cols, majority_files = sorted_groups[0]

        print(f"\n✅  MAJORITY STRUCTURE  ({len(majority_files)} file(s))")
        print(f"    Column count : {len(majority_cols)}")
        print(f"    Columns      : {', '.join(majority_cols)}")

        if len(sorted_groups) > 1:
            print(f"\n⚠️   DEVIATING STRUCTURES  ({len(sorted_groups) - 1} variant(s) found)")
            majority_set = set(majority_cols)

            for cols, files in sorted_groups[1:]:
                this_set = set(cols)
                missing  = sorted(majority_set - this_set)
                extra    = sorted(this_set - majority_set)

                print(f"\n    Column count : {len(cols)}")
                print(f"    Columns      : {', '.join(cols)}")
                if missing:
                    print(f"    ─ Missing vs majority : {', '.join(missing)}")
                if extra:
                    print(f"    + Extra vs majority   : {', '.join(extra)}")
                print(f"    Affected files ({len(files)}):")
                for f in files:
                    state = infer_state(f)
                    year_folder = ""
                    for part in f.parts:
                        if "standards" in part.lower() or part.isdigit():
                            year_folder = f"  [{part}]"
                            break
                    print(f"      [{state}]{year_folder}  {f.name}")
        else:
            print("\n✅  All readable CSVs share the same column structure.")

    if unreadable:
        print(f"\n❌  UNREADABLE FILES  ({len(unreadable)})")
        for f in unreadable:
            print(f"    {f}")

    if excluded:
        print(f"\n⏭️   EXCLUDED FILES  ({len(excluded)})  — matched EXCLUDE_PATH_FRAGMENTS")
        for f in excluded:
            print(f"    {f}")

    if unknown_folders:
        print(f"\n🔍  UNKNOWN LEARNOSITY FOLDERS  ({len(unknown_folders)})")
        print(    "    These contain 'Learnosity' in the name but don't match any known")
        print(    "    suffix. They were NOT scanned. Add the suffix to")
        print(    "    LEARNOSITY_FOLDER_SUFFIXES if they should be included.")
        for folder in unknown_folders:
            print(f"    {folder}")

    print(f"\n{'=' * 64}\n")


def combine_csvs(csv_files: list[Path], output_path: Path) -> pd.DataFrame:
    """
    Load all provided CSVs and concatenate them into one DataFrame.
    Adds three provenance columns: 'state', 'standards_year', 'source_file'.
    All columns read as strings to prevent silent type coercion.
    """
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig", dtype=str)
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding="latin-1", dtype=str)

        df.insert(0, "state", infer_state(f))
        df.insert(1, "standards_year", infer_standards_year(f))
        df.insert(2, "source_file", f.name)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅  Combined CSV saved to: {output_path}")
    print(f"    Rows: {len(combined):,}  |  Columns: {len(combined.columns)}")
    return combined


def main():
    root = Path(ROOT_DIR)

    if ROOT_DIR == "PASTE_YOUR_SYNCED_PATH_HERE":
        print("ERROR: Set ROOT_DIR at the top of the script before running.")
        sys.exit(1)

    if not root.exists():
        print(f"ERROR: Path not found — {root}")
        print("Check that the OneDrive sync is complete and ROOT_DIR is correct.")
        sys.exit(1)

    print(f"\nScanning: {root}")
    csv_files, excluded = find_learnosity_csvs(root)
    unknown_folders = find_unknown_learnosity_folders(root)
    print(f"Found {len(csv_files)} CSV(s) in known Learnosity folders.")
    if excluded:
        print(f"Skipped {len(excluded)} CSV(s) due to EXCLUDE_PATH_FRAGMENTS.")
    if unknown_folders:
        print(f"⚠️  Found {len(unknown_folders)} unrecognised Learnosity folder(s) — see report.")

    if not csv_files:
        print("Nothing to process. Confirm the sync finished and folder names match.")
        sys.exit(0)

    audit_result = audit(csv_files)
    print_audit_report(audit_result, excluded, unknown_folders)

    groups = audit_result["structure_groups"]
    if not groups:
        print("No readable files to combine. Exiting.")
        sys.exit(0)

    sorted_groups   = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    majority_cols, majority_files = sorted_groups[0]
    deviating_count = sum(len(v) for _, v in sorted_groups[1:])

    if deviating_count > 0:
        print(
            f"Note: {deviating_count} file(s) with deviating structures will be "
            f"EXCLUDED from the combined output by default.\n"
            f"Review the report above before deciding how to handle them.\n"
        )

    response = input(
        f"Combine {len(majority_files)} standard-structure file(s) into "
        f"'{OUTPUT_CSV}'? [y/N]: "
    )
    if response.strip().lower() != "y":
        print("Aborted — no files written.")
        sys.exit(0)

    combine_csvs(majority_files, Path(OUTPUT_CSV))


if __name__ == "__main__":
    main()