#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:?Usage: $0 /path/to/lanl_cyber1 /path/to/output [llama_model_dir] [ner_model_dir]}"
OUT_DIR="${2:?Usage: $0 /path/to/lanl_cyber1 /path/to/output [llama_model_dir] [ner_model_dir]}"
LLAMA_MODEL_DIR="${3:-}"
NER_MODEL_DIR="${4:-}"

mkdir -p "$OUT_DIR"

python -m src.lanl_proxy \
  --data-dir "$DATA_DIR" \
  --output "$OUT_DIR/lanl_proxy_100k.csv.gz" \
  --sample-size 100000 \
  --positive-size 5000 \
  --seed 42

python -m src.split_data \
  --input "$OUT_DIR/lanl_proxy_100k.csv.gz" \
  --train-output "$OUT_DIR/real_train.csv.gz" \
  --test-output "$OUT_DIR/real_test.csv.gz" \
  --test-size 0.2 \
  --seed 42 \
  --mode stratified

DEID_ARGS=(--proxy "$OUT_DIR/real_test.csv.gz" --output "$OUT_DIR/deid_195_results.json")
if [[ -n "$NER_MODEL_DIR" ]]; then
  DEID_ARGS+=(--ner-model "$NER_MODEL_DIR")
fi
python -m src.deid_benchmark "${DEID_ARGS[@]}"

# Train CTGAN on REAL TRAIN ONLY. 19:1 is preserved exactly by class-conditional synthesis.
TRAIN_POS=$((5000 * 80 / 100))
TRAIN_NEG=$((95000 * 80 / 100))
python -m src.synthesize_sdv \
  --input-train "$OUT_DIR/real_train.csv.gz" \
  --output "$OUT_DIR/lanl_ctgan_train.csv.gz" \
  --epochs 300 \
  --seed 42 \
  --positive-size "$TRAIN_POS" \
  --negative-size "$TRAIN_NEG"

EVAL_ARGS=(
  --real-train "$OUT_DIR/real_train.csv.gz"
  --real-test "$OUT_DIR/real_test.csv.gz"
  --ctgan "$OUT_DIR/lanl_ctgan_train.csv.gz"
  --output "$OUT_DIR/model_metrics.json"
  --seed 42
)

if [[ -n "$LLAMA_MODEL_DIR" ]]; then
  python -m src.synthesize_llama \
    --input-train "$OUT_DIR/real_train.csv.gz" \
    --output "$OUT_DIR/lanl_llama3_train.csv.gz" \
    --model-dir "$LLAMA_MODEL_DIR" \
    --rows $((TRAIN_POS + TRAIN_NEG)) \
    --positive-rows "$TRAIN_POS" \
    --batch-rows 32 \
    --seed 42
  EVAL_ARGS+=(--llama "$OUT_DIR/lanl_llama3_train.csv.gz")
fi

python -m src.evaluate "${EVAL_ARGS[@]}"

python -m src.association_analysis \
  --input "$OUT_DIR/real_train.csv.gz" \
  --output "$OUT_DIR/association_analysis.json"

python -m src.paper_tables \
  --metrics "$OUT_DIR/model_metrics.json" \
  --deid "$OUT_DIR/deid_195_results.json" \
  --output "$OUT_DIR/paper_tables.md"

echo "Done. Do not update the manuscript until paper_tables.md and split_meta.json have been checked against the actual run."
