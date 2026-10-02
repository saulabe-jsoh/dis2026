#!/usr/bin/env python3
"""samples/*.txt 판결문에 v1 사전(정확일치+정규식)을 적용 → 문서별·분류별 히트 집계 CSV.
사용: python analyze_hits.py --samples samples --dict dict/defense_public_dict_v1.csv --out results/hits_by_doc.csv
"""
import argparse, csv, re, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from benchmark import Matcher

ap = argparse.ArgumentParser()
ap.add_argument("--samples", default="samples"); ap.add_argument("--dict", default="dict/defense_public_dict_v1.csv")
ap.add_argument("--out", default="results/hits_by_doc.csv"); ap.add_argument("--top", default="results/top_terms.csv")
a = ap.parse_args()

m, meta, regs = Matcher(), {}, []
for r in csv.DictReader(open(a.dict, encoding="utf-8-sig")):
    t = unicodedata.normalize("NFC", r["term"])
    if r["match_type"] == "regex":
        regs.append((re.compile(t), r["category"], r["subcategory"])); continue
    m.add(t, 0); meta[t] = (r["category"], r["subcategory"], r["verify_status"])
m.finalize()

cats = sorted({v[0] for v in meta.values()} | {c for _, c, _ in regs})
rows, top = [], Counter()
for fp in sorted(Path(a.samples).glob("*.txt")):
    t = unicodedata.normalize("NFC", fp.read_text(encoding="utf-8", errors="ignore"))
    c = Counter()
    for st, en, _ in m.spans(t):
        term = t[st:en]; c[meta.get(term, ("?",))[0]] += 1; top[(term, meta.get(term, ("?",))[0])] += 1
    for rx, cat, _ in regs:
        c[cat + "(regex)"] += len(rx.findall(t))
    rows.append({"file": fp.name, "chars": len(t), "total_hits": sum(c.values()), **{k: c.get(k, 0) for k in cats}})
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["file"]); w.writeheader(); w.writerows(rows)
with open(a.top, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["term", "category", "hits", "legit(수기검수: Y/N)"])
    for (t, c), n in top.most_common(100): w.writerow([t, c, n, ""])
print(f"{len(rows)}건 분석 → {a.out}, {a.top}")
