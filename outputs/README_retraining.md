# 0525 retraining package

This package retrains the relation classifier with `Train_Set_clean_augmented`
and predicts only the 719 unresolved rows in
`repredict_719_package/pending_719_for_model.csv`.

## Files

- `Train_Set_clean_augmented/`: cleaned and augmented training data, one CSV per label.
- `repredict_719_package/pending_719_for_model.csv`: 719 rows to re-predict. Keep `RowID`.
- `submission_0.88455.csv`: current full 4068-row baseline submission.
- `baseline/train.py`: model training script.
- `baseline/infer.py`: model inference script, now also supports top-k probability export.
- `step1_prepare_0525.py`: validates the package and writes `labels_from_train_clean_augmented.txt`.
- `step2_train_8seeds.sh`: trains 8 seeds.
- `step3_infer_fuse_719.py`: runs inference for all trained seeds, fuses predictions, and writes outputs.

## Server usage

Run from the `0525` directory:

```bash
python3 step1_prepare_0525.py
bash step2_train_8seeds.sh
python3 step3_infer_fuse_719.py
```

If your server has different GPU ids:

```bash
GPU_IDS="0 1" bash step2_train_8seeds.sh
python3 step3_infer_fuse_719.py --gpu_id 0
```

Optional hyperparameter overrides:

```bash
BATCH_SIZE=64 EPOCHS=20 LR=3e-5 MAX_LENGTH=128 bash step2_train_8seeds.sh
```

## Main outputs

`step3_infer_fuse_719.py` writes timestamped files under `outputs_719/`:

- `pending_719_predicted_fused_*.csv`: 719-row prediction file with `Label` filled.
- `top30_candidates_long_fused_*.csv`: fused top-30 candidate table for later audit.
- `per_model_predictions_wide_*.csv`: each seed's top-1 prediction and probability.
- `submission_0.88455_repredict719_*.csv`: optional full 4068-row submission with the 719 rows replaced by fused predictions.

The full submission is for offline scoring/testing. For manual audit, use the
719-row fused file and the top-30 long file.
