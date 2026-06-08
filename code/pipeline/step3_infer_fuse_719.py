#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_model_paths(model_basedir: Path) -> list[Path]:
    return sorted(model_basedir.glob("seed_*/cpa_*/best_model.pdparams"))


def label_classes_for_model(model_path: Path, fallback: Path | None = None) -> Path:
    label_path = model_path.parent / "label_classes.txt"
    if label_path.exists():
        return label_path
    if fallback and fallback.exists():
        return fallback
    raise FileNotFoundError(f"No label_classes.txt beside {model_path}")


def model_tag(model_path: Path) -> str:
    return f"{model_path.parts[-3]}_{model_path.parts[-2]}"


def run_inference_for_model(args: argparse.Namespace, model_path: Path, labels_path: Path) -> tuple[Path, Path]:
    tag = model_tag(model_path)
    out_csv = args.pred_dir / f"pred_{tag}.csv"
    topk_csv = args.pred_dir / f"top{args.top_k}_{tag}.csv"
    if out_csv.exists() and topk_csv.exists() and not args.force:
        print(f"  [SKIP] {tag}: existing prediction files found")
        return out_csv, topk_csv

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if args.pythonpath_add:
        sep = ";" if os.name == "nt" else ":"
        env["PYTHONPATH"] = str(args.pythonpath_add) + sep + env.get("PYTHONPATH", "")

    cmd = [
        args.python_bin,
        str(args.infer_py),
        "--input_csv",
        str(args.input_csv),
        "--labels_path",
        str(labels_path),
        "--model_path",
        str(model_path),
        "--output_file",
        str(out_csv),
        "--top_k",
        str(args.top_k),
        "--topk_output_file",
        str(topk_csv),
        "--shortcut_name",
        args.shortcut_name,
        "--batch_size",
        str(args.batch_size),
        "--max_length",
        str(args.max_length),
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
    ]
    if args.use_amp:
        cmd.append("--use_amp")

    print(f"  Running inference: {tag} on CUDA_VISIBLE_DEVICES={args.gpu_id}")
    result = subprocess.run(cmd, env=env, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Inference failed for {model_path}: return code {result.returncode}")
    return out_csv, topk_csv


def validate_prediction_alignment(input_df: pd.DataFrame, pred_dfs: list[pd.DataFrame]) -> None:
    expected_rowids = input_df["RowID"].astype(int).tolist()
    for idx, df in enumerate(pred_dfs):
        if len(df) != len(input_df):
            raise RuntimeError(f"Prediction {idx} row count mismatch: {len(df)} vs {len(input_df)}")
        if "RowID" not in df.columns:
            raise RuntimeError(f"Prediction {idx} missing RowID column")
        rowids = df["RowID"].astype(int).tolist()
        if rowids != expected_rowids:
            raise RuntimeError(f"Prediction {idx} RowID order does not match input")
        if "Label" not in df.columns:
            raise RuntimeError(f"Prediction {idx} missing Label column")


def fuse_predictions(input_df: pd.DataFrame, pred_paths: list[Path], model_tags: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_dfs = [read_csv(p) for p in pred_paths]
    validate_prediction_alignment(input_df, pred_dfs)

    n_models = len(pred_dfs)
    labels_matrix = [df["Label"].astype(str).tolist() for df in pred_dfs]
    probs_matrix = []
    for df in pred_dfs:
        if "prediction_probability" in df.columns:
            probs_matrix.append(pd.to_numeric(df["prediction_probability"], errors="coerce").fillna(0.0).tolist())
        else:
            probs_matrix.append([0.0] * len(df))

    fused_labels: list[str] = []
    votes: list[int] = []
    vote_fraction: list[float] = []
    mean_probability: list[float] = []
    model_vote_strings: list[str] = []

    for row_idx in range(len(input_df)):
        row_labels = [labels_matrix[m][row_idx] for m in range(n_models)]
        row_probs = [float(probs_matrix[m][row_idx]) for m in range(n_models)]
        counter = Counter(row_labels)
        label_to_probs: dict[str, list[float]] = defaultdict(list)
        for label, prob in zip(row_labels, row_probs):
            label_to_probs[label].append(prob)

        ranked = sorted(
            counter.keys(),
            key=lambda label: (
                counter[label],
                float(np.mean(label_to_probs[label])) if label_to_probs[label] else 0.0,
                label,
            ),
            reverse=True,
        )
        best = ranked[0]
        fused_labels.append(best)
        votes.append(counter[best])
        vote_fraction.append(counter[best] / n_models)
        mean_probability.append(float(np.mean(label_to_probs[best])) if label_to_probs[best] else 0.0)
        model_vote_strings.append(
            " | ".join(f"{tag}:{label}" for tag, label in zip(model_tags, row_labels))
        )

    fused = input_df.copy()
    fused["Label"] = fused_labels
    fused["fused_label"] = fused_labels
    fused["votes"] = votes
    fused["n_models"] = n_models
    fused["vote_fraction"] = vote_fraction
    fused["mean_top1_probability_for_label"] = mean_probability
    fused["model_votes"] = model_vote_strings

    wide = input_df[["LocalID", "RowID", "Subject", "Object"]].copy()
    for tag, df in zip(model_tags, pred_dfs):
        wide[f"{tag}_label"] = df["Label"].astype(str).tolist()
        if "prediction_probability" in df.columns:
            wide[f"{tag}_probability"] = pd.to_numeric(
                df["prediction_probability"],
                errors="coerce",
            ).fillna(0.0).tolist()

    return fused, wide


def aggregate_topk(input_df: pd.DataFrame, topk_paths: list[Path], model_tags: list[str], top_k: int) -> pd.DataFrame:
    frames = []
    for tag, path in zip(model_tags, topk_paths):
        df = read_csv(path)
        df["model"] = tag
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    topk = pd.concat(frames, ignore_index=True)
    topk["RowID"] = topk["RowID"].astype(int)
    topk["probability"] = pd.to_numeric(topk["probability"], errors="coerce").fillna(0.0)

    top1 = topk[topk["rank"] == 1].groupby(["RowID", "candidate_label"]).size().rename("top1_votes")
    grouped = topk.groupby(["RowID", "candidate_label"], as_index=False).agg(
        support_models=("model", "nunique"),
        probability_sum=("probability", "sum"),
        mean_probability_supported=("probability", "mean"),
        max_probability=("probability", "max"),
        best_rank=("rank", "min"),
    )
    grouped = grouped.merge(top1.reset_index(), on=["RowID", "candidate_label"], how="left")
    grouped["top1_votes"] = grouped["top1_votes"].fillna(0).astype(int)
    n_models = len(model_tags)
    grouped["mean_probability_all_models"] = grouped["probability_sum"] / n_models
    grouped["top1_vote_fraction"] = grouped["top1_votes"] / n_models

    meta_cols = ["LocalID", "RowID", "Subject", "Object"]
    meta = input_df[meta_cols].copy()
    grouped = grouped.merge(meta, on="RowID", how="left")
    grouped = grouped.sort_values(
        ["RowID", "top1_votes", "mean_probability_all_models", "support_models", "max_probability"],
        ascending=[True, False, False, False, False],
    )
    grouped["rank"] = grouped.groupby("RowID").cumcount() + 1
    grouped = grouped[grouped["rank"] <= top_k].copy()
    first_cols = [
        "LocalID",
        "RowID",
        "Subject",
        "Object",
        "rank",
        "candidate_label",
        "top1_votes",
        "top1_vote_fraction",
        "support_models",
        "mean_probability_all_models",
        "mean_probability_supported",
        "max_probability",
        "best_rank",
    ]
    return grouped[first_cols]


def write_full_submission(baseline_path: Path, fused: pd.DataFrame, output_path: Path) -> None:
    baseline = read_csv(baseline_path)
    required = {"Subject", "Object", "Label"}
    if not required.issubset(baseline.columns):
        raise RuntimeError(f"{baseline_path} must contain {sorted(required)}")
    for row in fused.itertuples(index=False):
        baseline.at[int(row.RowID), "Label"] = row.Label
    baseline.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--model_basedir", type=Path, default=ROOT / "cpa_output_clean_aug")
    parser.add_argument("--input_csv", type=Path, default=ROOT / "repredict_719_package" / "pending_719_for_model.csv")
    parser.add_argument("--baseline_submission", type=Path, default=ROOT / "submission_0.88455.csv")
    parser.add_argument("--pred_dir", type=Path, default=ROOT / "predictions_719")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs_719")
    parser.add_argument("--infer_py", type=Path, default=ROOT / "baseline" / "infer.py")
    parser.add_argument("--labels_fallback", type=Path, default=ROOT / "labels_from_train_clean_augmented.txt")
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--pythonpath_add", type=str, default="")
    parser.add_argument("--shortcut_name", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--use_amp", dest="use_amp", action="store_true")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
    parser.add_argument("--write_full_submission", dest="write_full_submission", action="store_true")
    parser.add_argument("--no_full_submission", dest="write_full_submission", action="store_false")
    parser.set_defaults(use_amp=True, write_full_submission=True)
    args = parser.parse_args()

    args.pred_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_df = read_csv(args.input_csv)
    required_input = {"LocalID", "RowID", "Subject", "Object", "Label"}
    missing = required_input - set(input_df.columns)
    if missing:
        raise RuntimeError(f"{args.input_csv} missing columns: {sorted(missing)}")
    if len(input_df) != 719:
        raise RuntimeError(f"Expected 719 pending rows, got {len(input_df)}")

    model_paths = find_model_paths(args.model_basedir)
    if not model_paths:
        raise RuntimeError(f"No trained models found under {args.model_basedir}/seed_*/cpa_*/")
    print(f"Found {len(model_paths)} models")
    for path in model_paths:
        print(f"  {path}")

    pred_paths: list[Path] = []
    topk_paths: list[Path] = []
    tags: list[str] = []
    for model_path in model_paths:
        tag = model_tag(model_path)
        labels_path = label_classes_for_model(model_path, args.labels_fallback)
        pred_path, topk_path = run_inference_for_model(args, model_path, labels_path)
        pred_paths.append(pred_path)
        topk_paths.append(topk_path)
        tags.append(tag)

    fused, wide = fuse_predictions(input_df, pred_paths, tags)
    topk_fused = aggregate_topk(input_df, topk_paths, tags, args.top_k)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fused_path = args.output_dir / f"pending_719_predicted_fused_{ts}.csv"
    wide_path = args.output_dir / f"per_model_predictions_wide_{ts}.csv"
    topk_path = args.output_dir / f"top{args.top_k}_candidates_long_fused_{ts}.csv"
    report_path = args.output_dir / f"repredict_719_report_{ts}.json"
    full_submission_path = args.output_dir / f"submission_0.88455_repredict719_{ts}.csv"

    fused.to_csv(fused_path, index=False, encoding="utf-8-sig")
    wide.to_csv(wide_path, index=False, encoding="utf-8-sig")
    topk_fused.to_csv(topk_path, index=False, encoding="utf-8-sig")

    if args.write_full_submission:
        write_full_submission(args.baseline_submission, fused, full_submission_path)

    label_dist = fused["Label"].value_counts().rename_axis("Label").reset_index(name="Count")
    label_dist_path = args.output_dir / f"pending_719_label_distribution_{ts}.csv"
    label_dist.to_csv(label_dist_path, index=False, encoding="utf-8-sig")

    summary = {
        "timestamp": ts,
        "input_csv": str(args.input_csv),
        "baseline_submission": str(args.baseline_submission),
        "model_basedir": str(args.model_basedir),
        "models": len(model_paths),
        "rows": int(len(fused)),
        "top_k": int(args.top_k),
        "fused_prediction_csv": str(fused_path),
        "per_model_wide_csv": str(wide_path),
        "fused_topk_long_csv": str(topk_path),
        "label_distribution_csv": str(label_dist_path),
        "full_submission_csv": str(full_submission_path) if args.write_full_submission else None,
    }
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
