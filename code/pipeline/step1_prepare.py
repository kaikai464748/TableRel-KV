#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "Train_Set_clean_augmented"
PENDING_CSV = ROOT / "repredict_719_package" / "pending_719_for_model.csv"
BASELINE_SUBMISSION = ROOT / "submission_0.88455.csv"
OUT_LABELS = ROOT / "labels_from_train_clean_augmented.txt"
SUMMARY_JSON = ROOT / "prepare_0525_summary.json"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def main() -> None:
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Missing train dir: {TRAIN_DIR}")
    if not PENDING_CSV.exists():
        raise FileNotFoundError(f"Missing pending csv: {PENDING_CSV}")
    if not BASELINE_SUBMISSION.exists():
        raise FileNotFoundError(f"Missing baseline submission: {BASELINE_SUBMISSION}")

    train_files = sorted(TRAIN_DIR.glob("*.csv"))
    if not train_files:
        raise RuntimeError(f"No CSV files found under {TRAIN_DIR}")

    train_rows = 0
    bad_train_files: list[str] = []
    label_counts: list[dict[str, object]] = []
    for csv_path in train_files:
        df = read_csv(csv_path)
        df.columns = [str(c).strip() for c in df.columns]
        if not {"Subject", "Object"}.issubset(df.columns):
            bad_train_files.append(csv_path.name)
            continue
        clean_df = df[["Subject", "Object"]].dropna()
        train_rows += len(clean_df)
        label_counts.append({"Label": csv_path.stem, "Rows": len(clean_df)})

    if bad_train_files:
        raise RuntimeError(f"Train CSVs missing Subject/Object columns: {bad_train_files[:20]}")

    labels = [p.stem for p in train_files]
    OUT_LABELS.write_text("\n".join(labels) + "\n", encoding="utf-8")
    pd.DataFrame(label_counts).sort_values("Label").to_csv(
        ROOT / "train_clean_augmented_label_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pending = read_csv(PENDING_CSV)
    pending.columns = [str(c).strip() for c in pending.columns]
    required_pending = {"RowID", "Subject", "Object", "Label"}
    missing_pending = required_pending - set(pending.columns)
    if missing_pending:
        raise RuntimeError(f"pending_719_for_model.csv missing columns: {sorted(missing_pending)}")
    if pending["RowID"].duplicated().any():
        raise RuntimeError("pending_719_for_model.csv has duplicated RowID values")
    if len(pending) != 719:
        raise RuntimeError(f"Expected 719 pending rows, got {len(pending)}")

    baseline = read_csv(BASELINE_SUBMISSION)
    baseline.columns = [str(c).strip() for c in baseline.columns]
    if not {"Subject", "Object", "Label"}.issubset(baseline.columns):
        raise RuntimeError("submission_0.88455.csv must contain Subject,Object,Label")
    if pending["RowID"].max() >= len(baseline) or pending["RowID"].min() < 0:
        raise RuntimeError("RowID values are outside baseline submission row range")

    summary = {
        "root": str(ROOT),
        "train_dir": str(TRAIN_DIR),
        "train_files": len(train_files),
        "train_rows": int(train_rows),
        "labels": len(labels),
        "pending_csv": str(PENDING_CSV),
        "pending_rows": int(len(pending)),
        "baseline_submission": str(BASELINE_SUBMISSION),
        "baseline_rows": int(len(baseline)),
        "labels_txt": str(OUT_LABELS),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
