"""Quick integrity/schema checks for the five LANL Cyber1 gzip sources."""
from __future__ import annotations
from pathlib import Path
import argparse
import gzip
import json
import pandas as pd

SOURCES = {
    "auth.txt.gz": 9,
    "proc.txt.gz": 5,
    "flows.txt.gz": 9,
    "dns.txt.gz": 3,
    "redteam.txt.gz": 4,
}


def inspect(path: Path, expected_cols: int, sample_rows: int = 5):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        rows = []
        for i, line in enumerate(fh):
            rows.append(line.rstrip("\n"))
            if i + 1 >= sample_rows:
                break
    field_counts = [len(x.split(",")) for x in rows]
    return {
        "exists": path.exists(),
        "bytes_compressed": path.stat().st_size if path.exists() else None,
        "sample_field_counts": field_counts,
        "expected_columns": expected_cols,
        "sample_ok": all(x == expected_cols for x in field_counts),
        "sample_rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {name: inspect(args.data_dir / name, n) for name, n in SOURCES.items()}
    bad = [k for k, v in report.items() if not v["exists"] or not v["sample_ok"]]
    report["status"] = "PASS" if not bad else "FAIL"
    report["bad_sources"] = bad
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(1 if bad else 0)

if __name__ == "__main__":
    main()
