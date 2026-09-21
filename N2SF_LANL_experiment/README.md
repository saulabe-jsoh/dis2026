# N2SF-LANL reproducible experiment package

This package rewrites the experiment so that the **paper, data split, synthesis models, and evaluation are mutually consistent**.

## Why the previous experiment needed correction

The manuscript said `79,000 train / 21,000 test`, while a 20% split of 100,000 rows yields 80,000/20,000. The old confusion matrix also used `20,000 normal + 1,000 threat`, which cannot coexist with an exact 95/5 100k dataset and a 20% test split. The old GitHub code also contained a synthetic/noise fallback and did not run real CTGAN/Llama-3 generation in the same path as the reported TSTR table. This package removes those inconsistencies.

## Experimental design used here

### Dataset

- Five official LANL Cyber1 files: `auth.txt.gz`, `proc.txt.gz`, `flows.txt.gz`, `dns.txt.gz`, `redteam.txt.gz`.
- `auth.txt.gz` is the anchor stream because LANL describes redteam events as known bad-behavior events taken from authentication data.
- Positive label: exact match on `(time, source user, source computer, destination computer)` with a RedTeam event.
- Negative label: authentication event not matched to RedTeam.
- The 100,000-row proxy dataset is deliberately sampled to 95,000 negatives + 5,000 positives. **This is an experimental sampling ratio, not the natural LANL attack prevalence and not a claim about actual defense networks.**

### Train/test split

A single stratified 80/20 split is created once and reused everywhere:

- REAL TRAIN: 80,000 rows = 76,000 normal + 4,000 threat
- REAL TEST: 20,000 rows = 19,000 normal + 1,000 threat

The exact counts are written to `split_meta.json` and `model_metrics.json`.

For a stronger publication design, run an additional chronological split as a sensitivity analysis; do not silently mix split strategies.

### Multi-source feature construction

Eight features match the manuscript:

Categorical: `src_user, dst_user, src_comp, dst_comp, process_name, dns_query`

Numeric: `pkt_count, byte_count`

The script attaches process/DNS/flow information by deterministic exact-second keys and bounded aggregation. It avoids many-to-many merge explosion.

### De-identification benchmark

The public LANL Cyber1 corpus is already de-identified according to the official data description. Therefore this package does **not** claim that LANL naturally contains 195 PII instances. Instead, it injects exactly 195 predefined sensitive-information targets into the REAL TEST set and evaluates Regex-only versus Regex+NER. The 195 target types are fixed in code before evaluation.

### Synthesis

`src/synthesize_sdv.py` performs **real class-conditional SDV-CTGAN synthesis** on REAL TRAIN ONLY.

`src/synthesize_llama.py` performs **real local Llama-3 generation** on REAL TRAIN ONLY. It uses `local_files_only=True`; no external API or automatic model download is allowed.

The Llama script is strict: invalid JSON or schema-invalid rows are rejected and retried. There is no Gaussian-noise or label-flip substitute.

### Evaluation

Every model is tested on the identical REAL TEST set.

- TRTR: real train -> XGBoost -> real test
- TSTR-CTGAN: synthetic train -> XGBoost -> real test
- TSTR-Llama-3: synthetic train -> XGBoost -> real test

Metrics:

- Precision
- Recall
- F1-Score
- FPR
- FNR
- PR-AUC
- ROC-AUC
- Confusion matrix

PR-AUC is computed from predicted probabilities, so it cannot be inferred from the confusion matrix alone.

## Installation

```
pip install -r requirements.txt
```

For N2SF/offline use, pre-stage the NER and Llama model directories locally. The model-loading code is deliberately configured not to reach the public model hub.

## First check

```
python self_test.py
```

This verifies only the arithmetic of the old manuscript's `TN=20,000, FP=0, FN=119, TP=881` example. It does **not** verify that those counts came from the actual LANL experiment.

## Full run

```
./run_reproducible_experiment.sh /path/to/lanl /path/to/output /path/to/local/llama3 /path/to/local/ner
```

For CTGAN-only run, omit the third argument. For Regex-only benchmark debugging, omit the fourth argument.

## Files that matter for the paper

- `output/split_meta.json` - actual train/test class counts
- `output/deid_195_results.json` - actual 195-target benchmark
- `output/model_metrics.json` - final TRTR/TSTR metrics and confusion matrices
- `output/association_analysis.json` - categorical association analysis without numeric LabelEncoder artifacts
- `output/paper_tables.md` - tables that can be copied into the manuscript after manual checking

## Publication rule

Never preserve the old values `0.9848`, `0.9390`, `0.9367`, `20,000/1,000`, or `195/195` merely because they appear in the previous draft. The new manuscript should contain the values produced by this reproducible pipeline and the actual experimental logs.

## LANL-specific caution for the manuscript

The official LANL description shows that processes and resolved hosts are de-identified (for example `P16`, `P25`, `C2109`). Therefore do not describe strings such as `c2-server.net` or `powershell.exe` as native LANL attack signatures unless those strings came from a documented additional preprocessing step. Those examples appeared in an earlier synthetic fallback and should not be carried into the empirical-data description.
