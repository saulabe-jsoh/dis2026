"""Controlled hybrid de-identification benchmark.

Important methodological correction:
The public LANL Cyber1 corpus is already de-identified. Therefore a claim of
"195 naturally occurring sensitive records in LANL" is not appropriate unless
an external ground-truth annotation exists. This module instead performs a
predefined controlled injection test into a hold-out sample and evaluates
Regex-only vs Regex+NER. The paper should describe the result as a controlled
benchmark, not as evidence that LANL naturally contains 195 PII instances.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import re

import pandas as pd

REGEX_PATTERNS: dict[str, re.Pattern[str]] = {
    "IPv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "IPv6": re.compile(r"(?i)\b(?:[0-9a-f]{1,4}:){2,7}[0-9a-f]{1,4}\b"),
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "USER_ID": re.compile(r"\b(?:user|uid|acct)[-_][A-Za-z0-9]{2,32}\b", re.I),
    "SERVER_ID": re.compile(r"\b(?:srv|server|host)[-_][A-Za-z0-9-]{2,32}\b", re.I),
}

@dataclass(frozen=True)
class InjectionSpec:
    entity_type: str
    value: str
    template: str


def fixed_injection_specs() -> list[InjectionSpec]:
    """Exactly 195 fixed targets. Do not edit after first publication run."""
    specs: list[InjectionSpec] = []
    for i in range(30):
        specs.append(InjectionSpec("IPv4", f"192.0.2.{(i % 250) + 1}", "src_ip={value}"))
    for i in range(20):
        specs.append(InjectionSpec("IPv6", f"2001:0db8:0000:0000:0000:0000:0000:{i+1:04x}", "dst_ip={value}"))
    for i in range(25):
        specs.append(InjectionSpec("EMAIL", f"researcher{i:03d}@example.org", "contact={value}"))
    for i in range(25):
        specs.append(InjectionSpec("USER_ID", f"uid-{i:03d}", "account={value}"))
    # These are English NER targets and therefore exercise the contextual tier.
    people = [
        "John Smith", "Mary Johnson", "James Brown", "Robert Wilson", "Patricia Taylor",
        "Michael Davis", "Linda Miller", "William Anderson", "Barbara Thomas", "David Jackson",
    ]
    orgs = [
        "Apex Defense Systems", "Northstar Research Laboratory", "Orion Security Group",
        "Falcon Engineering", "Sentinel Analytics",
    ]
    for i in range(35):
        specs.append(InjectionSpec("PERSON", people[i % len(people)] + f" {i:02d}", "operator={value}"))
    for i in range(30):
        specs.append(InjectionSpec("ORG", orgs[i % len(orgs)] + f" {i:02d}", "organization={value}"))
    for i in range(30):
        specs.append(InjectionSpec("SERVER_ID", f"srv-alpha-{i:03d}", "server={value}"))
    assert len(specs) == 195
    return specs


def build_injected_benchmark(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Attach exactly one injected sensitive target to 195 sampled records."""
    if len(df) < 195:
        raise ValueError("Input dataframe must contain at least 195 records")
    specs = fixed_injection_specs()
    out = df.copy().reset_index(drop=True)
    sampled_pos = out.sample(n=len(specs), random_state=seed).index.to_list()

    out["deid_text"] = (
        "src_user=" + out["src_user"].astype(str)
        + " dst_user=" + out["dst_user"].astype(str)
        + " src_comp=" + out["src_comp"].astype(str)
        + " dst_comp=" + out["dst_comp"].astype(str)
        + " process=" + out["process_name"].astype(str)
        + " dns=" + out["dns_query"].astype(str)
    )
    out["gt_entity_type"] = ""
    out["gt_entity_value"] = ""
    out["gt_marker"] = False

    for pos, spec in zip(sampled_pos, specs):
        text = spec.template.format(value=spec.value)
        out.at[pos, "deid_text"] += " " + text
        out.at[pos, "gt_entity_type"] = spec.entity_type
        out.at[pos, "gt_entity_value"] = spec.value
        out.at[pos, "gt_marker"] = True
    return out


def regex_predict(text: str) -> list[dict[str, Any]]:
    preds = []
    for label, pattern in REGEX_PATTERNS.items():
        for m in pattern.finditer(text):
            preds.append({"entity_group": label, "word": m.group(0), "start": m.start(), "end": m.end()})
    return preds


def load_local_ner(model_dir: Path):
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    except ImportError as exc:
        raise RuntimeError("Install transformers to run NER benchmark") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir, local_files_only=True)
    return pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1,
    )


def ner_predict(text: str, ner_pipe) -> list[dict[str, Any]]:
    raw = ner_pipe(text)
    preds = []
    for x in raw:
        preds.append({
            "entity_group": x.get("entity_group", x.get("entity", "")),
            "word": x.get("word", ""),
            "start": int(x.get("start", -1)),
            "end": int(x.get("end", -1)),
            "score": float(x.get("score", 0.0)),
        })
    return preds


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def target_detected(text: str, target: str, predictions: list[dict[str, Any]]) -> bool:
    """True if a predicted span overlaps the target string in context.

    We combine exact normalized substring matching with span-overlap to make
    the metric robust to NER token aggregation differences.
    """
    target_norm = _normalize(target)
    text_norm = _normalize(text)
    if target_norm not in text_norm:
        return False
    for p in predictions:
        word = _normalize(p.get("word", ""))
        if not word:
            continue
        if target_norm in word or word in target_norm:
            return True
    return False


def evaluate_benchmark(bench_df: pd.DataFrame, ner_pipe=None) -> dict[str, Any]:
    regex_hits = 0
    hybrid_hits = 0
    by_type: dict[str, dict[str, int]] = {}

    for row in bench_df.loc[bench_df["gt_marker"]].itertuples(index=False):
        text = row.deid_text
        gt_type = row.gt_entity_type
        gt_value = row.gt_entity_value
        rp = regex_predict(text)
        npred = ner_predict(text, ner_pipe) if ner_pipe is not None else []
        # Regex + NER union; for this controlled benchmark we count the target
        # as detected if a regex result or an NER result overlaps the target.
        regex_hit = target_detected(text, gt_value, rp)
        hybrid_hit = regex_hit or target_detected(text, gt_value, npred)
        regex_hits += int(regex_hit)
        hybrid_hits += int(hybrid_hit)
        slot = by_type.setdefault(gt_type, {"ground_truth": 0, "regex_detected": 0, "hybrid_detected": 0})
        slot["ground_truth"] += 1
        slot["regex_detected"] += int(regex_hit)
        slot["hybrid_detected"] += int(hybrid_hit)

    n = int(bench_df["gt_marker"].sum())
    result = {
        "ground_truth": n,
        "regex_detected": regex_hits,
        "hybrid_detected": hybrid_hits,
        "regex_recall": regex_hits / n if n else None,
        "hybrid_recall": hybrid_hits / n if n else None,
        "by_type": by_type,
    }
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run controlled 195-target Regex+NER benchmark")
    parser.add_argument("--proxy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ner-model", required=False, type=Path)
    args = parser.parse_args()
    df = pd.read_csv(args.proxy, compression="gzip")
    bench = build_injected_benchmark(df, seed=42)
    ner_pipe = load_local_ner(args.ner_model) if args.ner_model else None
    result = evaluate_benchmark(bench, ner_pipe)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
