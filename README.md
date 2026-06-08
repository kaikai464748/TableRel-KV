# TableRel-KV: Few-Shot Table Relation Extraction with Knowledge Verification

This repository contains the core materials for **TableRel-KV**, a few-shot table relation extraction system with knowledge-based verification.

## Overview

The project follows the table-column semantic modeling idea from *Annotating Columns with Pre-trained Language Models* and extends a pre-trained language model classification pipeline for Subject/Object relation prediction under few-shot settings.

The system predicts semantic relations for Subject/Object entity pairs from 563 candidate relation labels. Key challenges include long-tail label distribution, over-absorption by high-frequency relations, limited recall for rare labels, and directional confusion such as `part of/has part` and `capital/capital of`.

## Method

- 数据层：清洗重复、冲突与异常样本，并结合 Wikidata claims 与 PID 映射补充可靠长尾样本。
- 模型层：基于 BERT 等预训练编码模型训练多 seed 分类器，输出 top-30/top-100 候选关系及概率分布。
- 融合层：通过多 seed 投票与候选融合提升预测稳定性，并筛出低置信、高风险样本。
- 校验层：结合 Wikidata 正反向 claims、网页证据与大模型审查机制，对易混淆关系进行校验纠偏。

## Key Results

- 原始训练数据：39,233 行
- 清洗增强后训练数据：61,358 行
- 覆盖标签数：563 类
- 最终模型得分：0.90457
- baseline：0.66135
- 相对提升：约 36.8%

Relative improvement:

```text
(0.90457 - 0.66135) / 0.66135 = 36.78%
```

## 目录结构

```text
.
├── code/
│   ├── baseline/                 # 预训练编码模型训练与推理脚本
│   ├── pipeline/                 # 数据准备、多 seed 训练、候选融合流程
│   └── audit/                    # Wikidata / SPARQL / 搜索增强审计脚本
├── data/
│   ├── Train_Set_clean_augmented/ # 清洗增强后的 563 类训练数据
│   ├── labels.txt
│   ├── train_clean_augmented_label_counts.csv
│   ├── prepare_0525_summary.json
│   └── FINAL_SUMMARY.json
├── outputs/
│   ├── README_0525_retraining.md
│   ├── repredict_719_report.json
│   ├── wikidata_enhanced_round2_summary.json
│   └── final_two_rescue_summary.json
└── papers/
    ├── Annotating Columns with Pre-trained Language Models.pdf
    ├── 论文解读.md
    └── related_baseline_repository_README.md
```

## 对应关系

面向 GitHub 开源社区整理高质量表格语义关系数据集，将原始训练数据由 39,233 行清洗增强至 61,358 行：

- `data/Train_Set_clean_augmented/`
- `data/FINAL_SUMMARY.json`
- `data/train_clean_augmented_label_counts.csv`

基于 BERT 等编码模型训练多 seed 分类器，输出 top-30/top-100 候选关系及概率分布：

- `code/baseline/train.py`
- `code/baseline/infer.py`
- `code/pipeline/step2_train_8seeds.sh`
- `code/pipeline/step3_infer_fuse_719.py`
- `outputs/repredict_719_report.json`

结合 Wikidata claims、PID 映射、网页证据与大模型审查机制进行校验纠偏：

- `code/audit/`
- `outputs/wikidata_enhanced_round2_summary.json`
- `outputs/final_two_rescue_summary.json`
