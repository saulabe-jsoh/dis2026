"""Offline Llama-3 tabular log synthesis.

This script is deliberately strict:
- local_files_only=True: no network call from the high-security zone
- no noise/label-flip fallback
- JSONL output is parsed and schema-validated
- invalid model outputs are rejected

For publication, fix the local model revision and report it in the methods.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import logging
import re
import time

import pandas as pd

LOG = logging.getLogger(__name__)
CAT = ["src_user", "dst_user", "src_comp", "dst_comp", "process_name", "dns_query"]
NUM = ["pkt_count", "byte_count"]
ALL = CAT + NUM + ["label"]


def load_local_llama(model_dir: Path):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install torch and transformers before running Llama synthesis") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def build_prompt(examples: list[dict], n_rows: int, label_target: int | None = None) -> str:
    examples_json = json.dumps(examples, ensure_ascii=False)
    label_note = ""
    if label_target is not None:
        label_note = f" The label field MUST be {label_target} for every generated row."
    return f"""You are generating de-identified cybersecurity event records for research.
Return ONLY a JSON array and nothing else. Every object must contain exactly these fields:
{ALL}.
Categorical identifiers should be synthetic tokens with patterns similar to examples.
Numeric fields pkt_count and byte_count must be non-negative integers.
label is binary 0 or 1.{label_note}
Do not copy any example row exactly.
Examples:
{examples_json}
Generate exactly {n_rows} rows.
"""


def _extract_json_array(text: str) -> list[dict]:
    text = text.strip()
    # remove optional markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("No JSON array found in model output")
    payload = text[start:end+1]
    obj = json.loads(payload)
    if not isinstance(obj, list):
        raise ValueError("JSON payload is not an array")
    return obj


def validate_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        raise ValueError("Empty generation")
    clean = []
    for r in rows:
        if set(r.keys()) != set(ALL):
            raise ValueError(f"Schema mismatch: keys={sorted(r.keys())}")
        item = {c: r[c] for c in ALL}
        for c in CAT:
            item[c] = str(item[c])
        for c in NUM:
            item[c] = int(float(item[c]))
            if item[c] < 0:
                raise ValueError(f"Negative numeric value in {c}")
        item["label"] = int(item["label"])
        if item["label"] not in (0, 1):
            raise ValueError("label must be 0 or 1")
        clean.append(item)
    return pd.DataFrame(clean, columns=ALL)


def generate_batches(input_train_csv_gz: Path, output_csv_gz: Path, model_dir: Path,
                    rows: int, batch_rows: int, seed: int, max_retries: int,
                    positive_rows: int | None = None) -> dict:
    import torch
    train = pd.read_csv(input_train_csv_gz, compression="gzip")
    if rows <= 0:
        raise ValueError("rows must be positive")
    if positive_rows is None:
        positive_rows = int(round(rows * float(train["label"].mean())))
    negative_rows = rows - positive_rows
    if positive_rows <= 0 or negative_rows <= 0:
        raise ValueError("positive_rows and negative_rows must both be > 0")

    examples = train[ALL].sample(n=min(8, len(train)), random_state=seed).to_dict(orient="records")
    tokenizer, model = load_local_llama(model_dir)

    def gen_class(target_label: int, target_rows: int, offset: int):
        generated: list[pd.DataFrame] = []
        done = 0
        retries = 0
        while done < target_rows:
            n = min(batch_rows, target_rows - done)
            prompt = build_prompt(examples, n, label_target=target_label)
            messages = [
                {"role": "system", "content": "You generate only valid JSON arrays for a cybersecurity tabular synthesis benchmark."},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max(512, n * 80),
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            try:
                batch = validate_rows(_extract_json_array(decoded))
                if len(batch) != n:
                    raise ValueError(f"Model returned {len(batch)} rows; requested {n}")
                if not (batch["label"] == target_label).all():
                    raise ValueError("Generated batch contains a label other than the requested class")
            except Exception as exc:
                retries += 1
                LOG.warning("Rejected Llama batch (%s); retry=%d", exc, retries)
                if retries > max_retries:
                    raise RuntimeError("Too many invalid Llama generations; inspect local model prompt/output.") from exc
                continue
            generated.append(batch)
            done += len(batch)
            LOG.info("Generated label=%d: %d/%d", target_label, done, target_rows)
        return pd.concat(generated, ignore_index=True), retries

    pos_df, retry_pos = gen_class(1, positive_rows, 1)
    neg_df, retry_neg = gen_class(0, negative_rows, 2)
    synthetic = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    output_csv_gz.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(output_csv_gz, index=False, compression="gzip")
    meta = {
        "rows": len(synthetic),
        "positive_rows": int(synthetic["label"].sum()),
        "negative_rows": int((synthetic["label"] == 0).sum()),
        "seed": seed,
        "model_dir": str(model_dir),
        "retries": retry_pos + retry_neg,
        "offline_local_files_only": True,
    }
    output_csv_gz.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-train", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=80_000)
    parser.add_argument("--batch-rows", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--positive-rows", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    print(json.dumps(generate_batches(args.input_train, args.output, args.model_dir, args.rows, args.batch_rows, args.seed, args.max_retries, args.positive_rows), indent=2))

if __name__ == "__main__":
    main()
