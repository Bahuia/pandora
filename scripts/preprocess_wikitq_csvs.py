#!/usr/bin/env python3
"""
Preprocess WikiTQ CSV files to convert dates, times, and numeric values
into Python-comparable formats.

Conversion rules:
1. Dates like "January 5, 2014" → "2014-01-05" (ISO format, comparable as strings)
2. Numeric values with commas like "10,887" → 10887 (float)
3. Numeric values with currency like "$10.8 billion" → 10800000000.0 (float)
4. Percentages like "59%" → 0.59 (float)
5. Time values like "0:00" or "5:30" → seconds as float
6. Mixed values like "60,351 (69.1%)" → 60351.0 (extract number)
7. Ordinals like "1." → 1.0

Usage:
    python scripts/preprocess_wikitq_csvs.py

Output:
    Overwrites original CSV files in data/wikitq/csv/
    Creates backup in data/wikitq/csv_backup/
"""

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import tqdm


# Month name to number mapping
MONTH_MAP = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12,
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10,
    'nov': 11, 'dec': 12
}


def clean_numeric(val):
    """
    Clean numeric string to a float.

    Examples:
    - "10,887" → 10887.0
    - "$10.8 billion" → 10800000000.0
    - "59%" → 0.59
    - "60,351 (69.1%)" → 60351.0
    - "1." → 1.0
    - "0.0411" → 0.0411
    - "5:19.35" → "5:19.35" (Time strings are skipped!)
    """
    if not isinstance(val, str):
        return val

    val = val.strip()
    if not val:
        return val

    # 🔴 FIX: Skip values containing colons (likely time formats like 5:19.35)
    # We want to preserve time strings so LLM can return them as answers.
    if ':' in val:
        return val

    # Extract the main number (first number found)
    # Handle patterns like "60,351 (69.1%)" or "1.2 million"
    match = re.match(r'^[^\d]*([\d,]+\.?\d*)', val)
    if not match:
        return val

    num_str = match.group(1).replace(',', '')
    try:
        num = float(num_str)
    except ValueError:
        return val

    # Handle billion, million, thousand suffixes
    suffix_match = re.search(r'(billion|million|thousand|k)', val, re.IGNORECASE)
    if suffix_match:
        suffix = suffix_match.group(1).lower()
        if suffix == 'billion':
            num *= 1e9
        elif suffix == 'million':
            num *= 1e6
        elif suffix == 'thousand' or suffix == 'k':
            num *= 1e3

    # Handle percentage
    if '%' in val:
        num /= 100.0

    return num


def clean_date(val):
    """
    Clean date string to ISO format (YYYY-MM-DD).

    Examples:
    - "January 5, 2014" → "2014-01-05"
    - "Jan 5, 2014" → "2014-01-05"
    - "5 January 2014" → "2014-01-05"
    - "2014-01-05" → "2014-01-05" (already ISO)
    - "October 5" → "XXXX-10-05" (year missing)
    """
    if not isinstance(val, str):
        return val

    val = val.strip()
    if not val:
        return val

    # Pattern 1: "January 5, 2014" or "Jan 5, 2014"
    match = re.match(
        r'^(January|February|March|April|May|June|July|August|September|October|November|December|'
        r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s*(\d{4})',
        val, re.IGNORECASE
    )
    if match:
        month = MONTH_MAP.get(match.group(1).lower(), 0)
        day = int(match.group(2))
        year = int(match.group(3))
        return f"{year:04d}-{month:02d}-{day:02d}"

    # Pattern 2: "5 January 2014" or "5 Jan 2014"
    match = re.match(
        r'^(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December|'
        r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})',
        val, re.IGNORECASE
    )
    if match:
        day = int(match.group(1))
        month = MONTH_MAP.get(match.group(2).lower(), 0)
        year = int(match.group(3))
        return f"{year:04d}-{month:02d}-{day:02d}"

    # Pattern 3: "October 5" (no year)
    match = re.match(
        r'^(January|February|March|April|May|June|July|August|September|October|November|December|'
        r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})',
        val, re.IGNORECASE
    )
    if match:
        month = MONTH_MAP.get(match.group(1).lower(), 0)
        day = int(match.group(2))
        return f"XXXX-{month:02d}-{day:02d}"

    # Pattern 4: Already ISO format "2014-01-05"
    if re.match(r'^\d{4}-\d{2}-\d{2}$', val):
        return val

    return val


def clean_time(val):
    """
    Clean time string to seconds (float).

    Examples:
    - "0:00" → 0.0
    - "5:30" → 330.0
    - "1:30:45" → 5445.0
    - "5h 29' 10\"" → 19750.0
    """
    if not isinstance(val, str):
        return val

    val = val.strip()
    if not val:
        return val

    # Pattern: "H:MM:SS" or "MM:SS" or "H:MM"
    match = re.match(r'^(\d{1,2}):(\d{2}):(\d{2})$', val)
    if match:
        h, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return float(h * 3600 + m * 60 + s)

    match = re.match(r'^(\d{1,2}):(\d{2})$', val)
    if match:
        m, s = int(match.group(1)), int(match.group(2))
        return float(m * 60 + s)

    # Pattern: "5h 29' 10\""
    match = re.match(r'^(\d+)h\s+(\d+)\'\s+(\d+)', val)
    if match:
        h, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return float(h * 3600 + m * 60 + s)

    return val


def detect_column_type(series):
    """
    Detect the type of a column based on its values.

    Returns: 'date', 'time', 'numeric', or 'mixed'
    """
    vals = series.dropna().astype(str).tolist()
    if not vals:
        return 'unknown'

    date_count = 0
    time_count = 0
    num_count = 0

    for val in vals[:10]:  # Check first 10 values
        if clean_date(val) != val:
            date_count += 1
        elif clean_time(val) != val:
            time_count += 1
        elif clean_numeric(val) != val:
            num_count += 1

    total = min(len(vals), 10)
    if date_count > total * 0.5:
        return 'date'
    elif time_count > total * 0.5:
        return 'time'
    elif num_count > total * 0.5:
        return 'numeric'

    return 'mixed'


def preprocess_csv(csv_path, backup_dir=None):
    """
    Preprocess a single CSV file.

    Args:
        csv_path: Path to CSV file
        backup_dir: Optional backup directory

    Returns:
        dict with preprocessing stats
    """
    stats = {
        'file': str(csv_path),
        'rows': 0,
        'cols': 0,
        'date_cols': 0,
        'time_cols': 0,
        'numeric_cols': 0,
        'conversions': 0
    }

    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as e:
        print(f"  Error reading {csv_path}: {e}")
        return stats

    stats['rows'] = len(df)
    stats['cols'] = len(df.columns)

    for col in df.columns:
        col_type = detect_column_type(df[col])

        if col_type == 'date':
            stats['date_cols'] += 1
            df[col] = df[col].apply(clean_date)
            stats['conversions'] += 1
        elif col_type == 'time':
            # 🔴 DISABLE Time conversion to preserve original format (e.g., "5:19.35")
            pass
        elif col_type == 'numeric':
            stats['numeric_cols'] += 1
            df[col] = df[col].apply(clean_numeric)
            stats['conversions'] += 1
        elif col_type == 'mixed':
            # For mixed columns, try to convert individual numeric values only
            new_vals = []
            converted = 0
            for val in df[col]:
                # Skip time-like values
                if ':' in str(val):
                    new_vals.append(val)
                    continue
                cleaned_num = clean_numeric(val)
                if cleaned_num != val:
                    new_vals.append(cleaned_num)
                    converted += 1
                else:
                    new_vals.append(val)
            df[col] = new_vals
            if converted > 0:
                stats['conversions'] += 1

    # Save backup if requested
    if backup_dir:
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csv_path, backup_dir / csv_path.name)

    # Save preprocessed CSV
    df.to_csv(csv_path, index=False)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Normalize WikiTableQuestions CSV values")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--csv-dir", type=Path, default=None)
    parser.add_argument("--backup-dir", type=Path, default=None)
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    csv_dir = args.csv_dir or data_root / "wikitq" / "csv"
    backup_dir = args.backup_dir or data_root / "wikitq" / "csv_backup"
    progress_file = data_root / "wikitq" / "preprocess_progress.json"

    print("=== WikiTQ CSV Preprocessing ===\n")

    # Create backup directory
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Load progress if exists
    processed = set()
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
        processed = set(progress.get('processed_files', []))
        print(f"Loaded progress: {len(processed)} files already processed")

    # Get all CSV files
    csv_files = list(csv_dir.rglob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    # Filter out already processed
    remaining = [f for f in csv_files if str(f) not in processed]
    print(f"Remaining to process: {len(remaining)}\n")

    if not remaining:
        print("All files already processed!")
        return

    # Process files
    all_stats = []
    for i, csv_file in enumerate(tqdm.tqdm(remaining, desc="Preprocessing")):
        stats = preprocess_csv(csv_file, backup_dir=backup_dir)
        all_stats.append(stats)
        processed.add(str(csv_file))

        # Save progress every 100 files
        if (i + 1) % 100 == 0:
            with open(progress_file, 'w') as f:
                json.dump({'processed_files': list(processed)}, f)

    # Final save
    with open(progress_file, 'w') as f:
        json.dump({'processed_files': list(processed)}, f)

    # Print summary
    total_conversions = sum(s['conversions'] for s in all_stats)
    total_date_cols = sum(s['date_cols'] for s in all_stats)
    total_time_cols = sum(s['time_cols'] for s in all_stats)
    total_numeric_cols = sum(s['numeric_cols'] for s in all_stats)

    print(f"\n=== Summary ===")
    print(f"Files processed: {len(all_stats)}")
    print(f"Date columns converted: {total_date_cols}")
    print(f"Time columns converted: {total_time_cols}")
    print(f"Numeric columns converted: {total_numeric_cols}")
    print(f"Total conversions: {total_conversions}")
    print(f"Backups saved to: {backup_dir}")


if __name__ == "__main__":
    main()
