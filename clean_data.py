"""
Reusable dataset cleaner for CSV files.
Usage:
    python clean_data.py
Requirements:
    pip install pandas
The script asks which cleaning steps to apply, then exports CSV or SQLite
and writes a cleaning report.
"""

from pathlib import Path
import sqlite3
import pandas as pd


def ask_choice(prompt, choices, default):
    """Ask until the user enters one of the allowed choices."""
    choices_text = "/".join(choices)
    while True:
        answer = input(f"{prompt} ({choices_text}) [default: {default}]: ").strip().lower()
        if not answer:
            return default
        if answer in choices:
            return answer
        print(f"Please enter one of: {choices_text}")


def snapshot(df):
    """Collect basic dataset statistics for the report."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_by_column": df.isna().sum(),
        "dtypes": df.dtypes.astype(str),
    }


def iqr_outlier_counts(df):
    """Count potential IQR outliers in numeric columns; does not remove them."""
    results = {}
    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            count = 0
        else:
            count = int(((series < q1 - 1.5 * iqr) |
                         (series > q3 + 1.5 * iqr)).sum())
        results[col] = count
    return results


def write_report(path, source_path, before, after, actions, outliers_before, outliers_after):
    lines = [
        "DATA CLEANING REPORT",
        "=" * 60,
        f"Input file: {source_path}",
        "",
        "1. DATASET SUMMARY",
        f"Before cleaning: {before['rows']} rows x {before['columns']} columns",
        f"After cleaning:  {after['rows']} rows x {after['columns']} columns",
        f"Duplicate rows before/after: {before['duplicate_rows']} / {after['duplicate_rows']}",
        f"Missing cells before/after:  {before['missing_cells']} / {after['missing_cells']}",
        "",
        "2. CLEANING OPERATIONS",
    ]
    lines += [f"- {action}" for action in actions] if actions else ["- No cleaning operations selected."]
    lines += [
        "",
        "3. MISSING VALUES BY COLUMN (BEFORE)",
        "Column | Missing count",
        "-" * 40,
    ]
    lines += [f"{col} | {int(count)}" for col, count in before["missing_by_column"].items()]
    lines += [
        "",
        "4. DATA TYPES (AFTER)",
        "Column | Data type",
        "-" * 40,
    ]
    lines += [f"{col} | {dtype}" for col, dtype in after["dtypes"].items()]
    lines += [
        "",
        "5. POTENTIAL OUTLIERS (IQR RULE; NOT AUTOMATICALLY REMOVED)",
        "Column | Before | After",
        "-" * 40,
    ]
    all_cols = list(dict.fromkeys(list(outliers_before) + list(outliers_after)))
    lines += [
        f"{col} | {outliers_before.get(col, 0)} | {outliers_after.get(col, 0)}"
        for col in all_cols
    ]
    lines += [
        "",
        "6. NOTES / LIMITATIONS",
        "- IQR flags are screening signals, not proof that a value is erroneous.",
        "- Missing-value imputation can distort a target column; inspect the target before imputing it.",
        "- Confirm domain-specific rules (e.g. floor number <= total floors) before changing or deleting records.",
        "- Record the reason for each decision and compare model performance before and after cleaning.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    input_text = input("Path to input CSV [train.csv]: ").strip()
    input_path = Path(input_text or "train.csv").expanduser()
    if not input_path.exists():
        print(f"File not found: {input_path}")
        print("Place the CSV in this folder or enter its full path.")
        return

    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        print(f"Could not read CSV: {exc}")
        return

    before = snapshot(df)
    outliers_before = iqr_outlier_counts(df)
    actions = []

    print(f"\nLoaded {len(df):,} rows and {len(df.columns)} columns.")
    print(f"Missing cells: {before['missing_cells']:,}; duplicate rows: {before['duplicate_rows']:,}")
    print("\nMissing values by column:")
    print(before["missing_by_column"][before["missing_by_column"] > 0].to_string()
          if before["missing_cells"] else "No missing values found.")

    # 1. Normalize column names
    if ask_choice("Standardize column names (trim spaces, lowercase, spaces -> underscores)?",
                  ["y", "n"], "y") == "y":
        df.columns = (
            df.columns.astype(str).str.strip().str.lower()
            .str.replace(r"\s+", "_", regex=True)
            .str.replace(r"[^a-z0-9_]", "", regex=True)
        )
        actions.append("Standardized column names (lowercase, underscores, removed punctuation).")

    # 2. Clean text whitespace; do not force category capitalization
    if ask_choice("Trim leading/trailing spaces in text values?", ["y", "n"], "y") == "y":
        text_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in text_cols:
            df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        actions.append("Trimmed leading/trailing whitespace in text columns.")

    # 3. Duplicate handling
    if ask_choice("Remove exact duplicate rows?", ["y", "n"], "y") == "y":
        n_before = len(df)
        df = df.drop_duplicates()
        removed = n_before - len(df)
        actions.append(f"Removed {removed} exact duplicate row(s).")

    # 4. Missing values
    missing_count = int(df.isna().sum().sum())
    if missing_count:
        method = ask_choice(
            "Handle missing values: leave / drop_rows / impute",
            ["leave", "drop_rows", "impute"], "leave"
        )
        if method == "drop_rows":
            n_before = len(df)
            df = df.dropna()
            actions.append(f"Dropped {n_before - len(df)} row(s) containing at least one missing value.")
        elif method == "impute":
            numeric_cols = df.select_dtypes(include="number").columns
            text_cols = df.select_dtypes(include=["object", "string", "category"]).columns
            imputed = []
            for col in numeric_cols:
                count = int(df[col].isna().sum())
                if count:
                    df[col] = df[col].fillna(df[col].median())
                    imputed.append(f"{col}: median ({count} value(s))")
            for col in text_cols:
                count = int(df[col].isna().sum())
                if count:
                    mode = df[col].mode(dropna=True)
                    if not mode.empty:
                        df[col] = df[col].fillna(mode.iloc[0])
                        imputed.append(f"{col}: mode ({count} value(s))")
                    # If a whole column is missing, leave it missing rather than invent a value.
            actions.append("Imputed missing values where possible: " + (", ".join(imputed) if imputed else "none needed") + ".")
        else:
            actions.append("Left missing values unchanged.")
    else:
        actions.append("No missing values were present in the loaded dataset.")

    # 5. Optional domain checks: flag only, never auto-correct
    print("\nOptional sanity checks (flag rows only; no automatic deletion).")
    flags = []
    lower_to_original = {str(c).lower(): c for c in df.columns}

    def find_col(candidates):
        for candidate in candidates:
            if candidate in lower_to_original:
                return lower_to_original[candidate]
        return None

    floor_col = find_col(["floor_number", "floor", "floor_no"])
    total_col = find_col(["total_floors", "number_of_floors"])
    age_col = find_col(["age_of_property", "property_age", "age"])
    price_col = find_col(["price_inr_lakhs", "price", "sale_price", "target"])
    area_cols = [find_col(["super_area_sqft", "super_area", "area_sqft", "area"]),
                 find_col(["carpet_area_sqft", "carpet_area"])]
    if floor_col and total_col:
        bad = df[floor_col].notna() & df[total_col].notna() & (df[floor_col] > df[total_col])
        flags.append(f"Rows where {floor_col} > {total_col}: {int(bad.sum())}")
    if age_col:
        bad = df[age_col].notna() & (df[age_col] < 0)
        flags.append(f"Rows with negative {age_col}: {int(bad.sum())}")
    if price_col:
        bad = df[price_col].notna() & (df[price_col] <= 0)
        flags.append(f"Rows with non-positive {price_col}: {int(bad.sum())}")
    for area_col in area_cols:
        if area_col:
            bad = df[area_col].notna() & (df[area_col] <= 0)
            flags.append(f"Rows with non-positive {area_col}: {int(bad.sum())}")
    if flags:
        print("\n".join("- " + x for x in flags))
        actions.append("Ran domain sanity checks (flags only): " + "; ".join(flags) + ".")
    else:
        print("No recognized columns for the built-in domain checks.")
        actions.append("Built-in domain checks were skipped because matching column names were not found.")

    after = snapshot(df)
    outliers_after = iqr_outlier_counts(df)

    output_format = ask_choice("Output format: csv / sqlite", ["csv", "sqlite"], "csv")
    default_stem = input_path.stem + "_cleaned"
    output_name = input(f"Output filename [ {default_stem}.{ 'csv' if output_format == 'csv' else 'db' } ]: ").strip()
    if not output_name:
        output_name = default_stem + (".csv" if output_format == "csv" else ".db")
    output_path = Path(output_name).expanduser()

    try:
        if output_format == "csv":
            if output_path.suffix.lower() != ".csv":
                output_path = output_path.with_suffix(".csv")
            df.to_csv(output_path, index=False)
        else:
            if output_path.suffix.lower() not in [".db", ".sqlite", ".sqlite3"]:
                output_path = output_path.with_suffix(".db")
            table_name = input("SQLite table name [cleaned_data]: ").strip() or "cleaned_data"
            # Keep the table name simple to avoid SQL identifier issues.
            table_name = "".join(ch for ch in table_name if ch.isalnum() or ch == "_")
            if not table_name:
                table_name = "cleaned_data"
            with sqlite3.connect(output_path) as conn:
                df.to_sql(table_name, conn, if_exists="replace", index=False)
            actions.append(f"Saved data in SQLite table '{table_name}'.")
    except Exception as exc:
        print(f"Could not save output: {exc}")
        return

    report_path = output_path.with_name(output_path.stem + "_cleaning_report.txt")
    write_report(report_path, str(input_path), before, after, actions, outliers_before, outliers_after)

    print("\nFinished.")
    print(f"Cleaned dataset: {output_path.resolve()}")
    print(f"Cleaning report: {report_path.resolve()}")
    print(f"Final shape: {len(df):,} rows x {len(df.columns)} columns")


if __name__ == "__main__":
    main()
