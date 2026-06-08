import argparse
import os

import numpy as np
import pandas as pd
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader
from tqdm import tqdm

from paddlenlp.transformers import AutoTokenizer, AutoModel


# ==========================================
# 1. Model definition (must match training)
# ==========================================
class CPAModel(nn.Layer):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)

        hidden_size = None
        if hasattr(self.encoder, 'config'):
            if isinstance(self.encoder.config, dict):
                hidden_size = self.encoder.config.get('hidden_size', None)
            else:
                hidden_size = getattr(self.encoder.config, 'hidden_size', None)

        if hidden_size is None and hasattr(self.encoder, 'embeddings') and hasattr(self.encoder.embeddings, 'word_embeddings'):
            hidden_size = self.encoder.embeddings.word_embeddings.weight.shape[-1]

        if hidden_size is None:
            raise ValueError('Unable to infer hidden_size automatically. Please check the pretrained model.')

        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        if isinstance(outputs, tuple):
            sequence_output = outputs[0]
        elif hasattr(outputs, 'last_hidden_state'):
            sequence_output = outputs.last_hidden_state
        else:
            sequence_output = outputs

        cls_embedding = sequence_output[:, 0, :]
        logits = self.classifier(self.dropout(cls_embedding))
        return logits


# ==========================================
# 2. Tokenization helper
# ==========================================
def encode_pair(tokenizer, text_a, text_b, max_length):
    """Handle text-pair tokenization across different PaddleNLP versions."""
    try:
        encoding = tokenizer(
            text=text_a,
            text_pair=text_b,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
        )
    except TypeError:
        try:
            encoding = tokenizer(
                text_a,
                text_b,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
            )
        except TypeError:
            encoding = tokenizer(
                text=text_a,
                text_pair=text_b,
                max_seq_len=max_length,
                pad_to_max_seq_len=True,
                truncation=True,
                return_attention_mask=True,
            )

    input_ids = encoding['input_ids']
    attention_mask = encoding.get('attention_mask', None)

    if attention_mask is None:
        seq_len = encoding.get('seq_len', len(input_ids))
        seq_len = min(seq_len, max_length)
        attention_mask = [1] * seq_len + [0] * (max_length - seq_len)

    return np.array(input_ids, dtype='int64'), np.array(attention_mask, dtype='int64')


# ==========================================
# 3. Single-table inference dataset
# ==========================================
class SingleTableInferenceDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.original_rows = []

        # Read the CSV file.
        df = pd.read_csv(csv_path, low_memory=False, encoding='utf-8-sig')

        # Normalize column names by trimming whitespace.
        df.columns = [str(col).strip() for col in df.columns]

        # Locate Subject and Object columns in a case-insensitive way.
        subject_col = None
        object_col = None
        for col in df.columns:
            if col.lower() == 'subject':
                subject_col = col
            elif col.lower() == 'object':
                object_col = col

        if subject_col is None or object_col is None:
            raise ValueError("The CSV file must contain 'Subject' and 'Object' columns (case-insensitive).")

        # Drop rows with missing Subject/Object values.
        temp_df = df[[subject_col, object_col]].dropna()
        for idx, row in temp_df.iterrows():
            self.samples.append((str(row[subject_col]), str(row[object_col])))
            self.original_rows.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subject_text, object_text = self.samples[idx]
        input_ids, attention_mask = encode_pair(
            self.tokenizer,
            subject_text,
            object_text,
            self.max_length,
        )
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'orig_idx': np.int64(idx),
        }


def collate_fn(samples):
    return {
        'input_ids': np.stack([s['input_ids'] for s in samples]).astype('int64'),
        'attention_mask': np.stack([s['attention_mask'] for s in samples]).astype('int64'),
        'orig_idx': np.array([s['orig_idx'] for s in samples], dtype='int64'),
    }


# ==========================================
# 4. Device helper
# ==========================================
def resolve_device(device_arg):
    """Try the requested device first, otherwise fall back to CPU."""
    try:
        custom_types = paddle.device.get_all_custom_device_type()
    except Exception:
        custom_types = []

    print(f'Available custom devices: {custom_types}')

    if device_arg:
        try:
            dev = paddle.set_device(device_arg)
            print(f'Using requested device: {dev}')
            return dev
        except Exception as e:
            print(f'Failed to set requested device {device_arg}: {e}')

    dev = paddle.set_device('cpu')
    print('Falling back to CPU.')
    return dev


# ==========================================
# 5. Inference pipeline
# ==========================================
def run_inference(args):
    device = resolve_device(args.device)

    # Load label mapping.
    with open(args.labels_path, 'r', encoding='utf-8-sig') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    id2label = {idx: label for idx, label in enumerate(classes)}

    # Initialize tokenizer and model.
    tokenizer = AutoTokenizer.from_pretrained(args.shortcut_name)
    model = CPAModel(args.shortcut_name, len(classes))

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f'Model file not found: {args.model_path}')

    state_dict = paddle.load(args.model_path)
    model.set_state_dict(state_dict)
    model.eval()

    # Load the dataset.
    dataset = SingleTableInferenceDataset(args.input_csv, tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        return_list=True,
    )

    print(f'Starting inference. Total valid rows: {len(dataset)}')
    predictions = [None] * len(dataset)
    prediction_probs = [None] * len(dataset)
    topk_records = []
    use_amp = args.use_amp and str(device) != 'cpu'
    top_k = max(0, min(args.top_k, len(classes)))

    with paddle.no_grad():
        for batch in tqdm(dataloader, desc='Running inference'):
            ids = paddle.to_tensor(batch['input_ids'], dtype='int64')
            mask = paddle.to_tensor(batch['attention_mask'], dtype='int64')
            orig_indices = batch['orig_idx'].tolist()

            if use_amp:
                with paddle.amp.auto_cast(enable=True):
                    logits = model(ids, mask)
            else:
                logits = model(ids, mask)

            probs = F.softmax(logits, axis=1)
            preds = paddle.argmax(probs, axis=1).numpy().tolist()
            top_probs = None
            top_indices = None
            if top_k > 0:
                top_probs_tensor, top_indices_tensor = paddle.topk(probs, k=top_k, axis=1)
                top_probs = top_probs_tensor.numpy().tolist()
                top_indices = top_indices_tensor.numpy().tolist()

            for idx_in_batch, pred_idx in enumerate(preds):
                original_position = orig_indices[idx_in_batch]
                predictions[original_position] = id2label[pred_idx]
                prediction_probs[original_position] = float(probs[idx_in_batch, pred_idx].numpy())
                if top_k > 0:
                    for rank, (label_idx, prob) in enumerate(
                        zip(top_indices[idx_in_batch], top_probs[idx_in_batch]),
                        start=1,
                    ):
                        topk_records.append(
                            {
                                'valid_position': original_position,
                                'rank': rank,
                                'candidate_label': id2label[int(label_idx)],
                                'probability': float(prob),
                            }
                        )

    # Reload the original CSV and attach predictions.
    original_df = pd.read_csv(args.input_csv, low_memory=False, encoding='utf-8-sig')
    original_df.columns = [str(col).strip() for col in original_df.columns]

    subject_col = None
    object_col = None
    for col in original_df.columns:
        if col.lower() == 'subject':
            subject_col = col
        elif col.lower() == 'object':
            object_col = col

    if subject_col is None or object_col is None:
        raise ValueError("The CSV file must contain 'Subject' and 'Object' columns (case-insensitive).")

    # Only rows with valid Subject/Object pairs receive predictions.
    valid_mask = original_df[subject_col].notna() & original_df[object_col].notna()
    valid_indices = original_df[valid_mask].index.tolist()

    original_df['Label'] = None
    original_df['prediction_probability'] = None
    for row_idx, pred_label in zip(valid_indices, predictions):
        original_df.loc[row_idx, 'Label'] = pred_label
    for row_idx, pred_prob in zip(valid_indices, prediction_probs):
        original_df.loc[row_idx, 'prediction_probability'] = pred_prob

    # Save the result without modifying the source file.
    original_df.to_csv(args.output_file, index=False, encoding='utf-8-sig')
    print(f'Inference completed. Results saved to: {args.output_file}')

    if args.topk_output_file and top_k > 0:
        topk_df = pd.DataFrame(topk_records)
        valid_index_df = original_df.loc[valid_indices].reset_index().rename(columns={'index': 'source_row_index'})
        passthrough_cols = [
            col for col in ['LocalID', 'RowID', subject_col, object_col]
            if col in valid_index_df.columns
        ]
        meta = valid_index_df[passthrough_cols].copy()
        meta.insert(0, 'valid_position', range(len(meta)))
        topk_df = topk_df.merge(meta, on='valid_position', how='left')
        first_cols = [
            col for col in ['LocalID', 'RowID', subject_col, object_col, 'rank', 'candidate_label', 'probability']
            if col in topk_df.columns
        ]
        topk_df = topk_df[first_cols + [col for col in topk_df.columns if col not in first_cols and col != 'valid_position']]
        topk_df.to_csv(args.topk_output_file, index=False, encoding='utf-8-sig')
        print(f'Top-{top_k} probabilities saved to: {args.topk_output_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, default="./test.csv")
    parser.add_argument('--labels_path', type=str, default="./labels.txt")
    parser.add_argument('--model_path', type=str, default="./cpa_output/cpa_20260422_191904/best_model.pdparams")
    parser.add_argument('--output_file', type=str, default='./submission.csv')
    parser.add_argument('--shortcut_name', type=str, default='bert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--topk_output_file', type=str, default='')
    args = parser.parse_args()
    run_inference(args)
