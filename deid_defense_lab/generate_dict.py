#!/usr/bin/env python3
"""방산·안보수사 사전 규모별(1만~10만 항목) CSV 생성기.

- 최소 단위 = 사전 항목(term) 1개
- 시드(실제 용어) 항목은 모든 단계에 항상 포함, 나머지는 합성(synthetic) 항목으로 채움
- 단계는 '중첩(nested)' 구조: 10k ⊂ 20k ⊂ ... ⊂ 100k  (규모 외 변수 통제)
- 합성 항목은 is_synthetic=1 로 표시 → 실제 판결문에서 합성 항목이 히트되면 '오탐 후보'

사용:  python generate_dict.py --out dict --seed 42
"""
import argparse
import csv
import itertools
import random
import unicodedata
from pathlib import Path

SYSTEMS = ["유도무기", "전차", "자주포", "함정", "잠수함", "전투기", "헬기", "무인기", "레이더", "소나",
           "전술통신", "위성", "미사일", "방공체계", "전자전", "지휘통제", "탄약", "장갑", "추진체", "항법"]
PARTS = ["탐색기", "신호처리기", "안테나", "구동장치", "제어기", "센서모듈", "추진기관", "열영상장비",
         "통신모듈", "암호모듈", "전원장치", "냉각장치", "관성항법장치", "펌웨어", "시뮬레이터", "시험장비"]
DOCS = ["설계도면", "소스코드", "시험성적서", "공정도", "제조사양서", "정비교범", "성능분석자료", "구매규격서",
        "개발계획서", "형상관리문서"]
CODE_PREFIX = ["DP", "SYS", "TECH", "EXP", "SEC", "PRJ", "ROC", "MIL"]
GRADES = ["", "", "", "비밀 ", "대외비 "]

CAT_OF = {"system_part_doc": ("TECH_DOC", "합성-체계부품문서", "high"),
          "code": ("CODE", "합성-과제코드", "mid")}


def norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()


def synth_pool(rng: random.Random):
    pool = []
    for s, p, d in itertools.product(SYSTEMS, PARTS, DOCS):
        g = rng.choice(GRADES)
        pool.append((norm(f"{g}{s} {p} {d}"), *CAT_OF["system_part_doc"]))
    for pre, yr, seq in itertools.product(CODE_PREFIX, range(2005, 2027), range(1, 600)):
        pool.append((f"{pre}-{yr}-{seq:04d}", *CAT_OF["code"]))
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dict")
    ap.add_argument("--seed-file", default="dict/defense_public_dict_v1.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", type=int, default=10_000)
    ap.add_argument("--stop", type=int, default=100_000)
    ap.add_argument("--step", type=int, default=10_000)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    rows, seen = [], set()
    with open(a.seed_file, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("match_type", "exact") != "exact":
                continue  # 정규식 행은 벤치마크(정확일치)에서 제외
            t = norm(r["term"])
            if t and t not in seen:
                seen.add(t)
                src = r.get("source_id") or r.get("source", "")
                rows.append([t, r["category"], r["subcategory"], r["sensitivity"], src, 0])
    n_seed = len(rows)

    pool = [x for x in synth_pool(rng) if x[0] not in seen]
    rng.shuffle(pool)
    need = a.stop - n_seed
    if len(pool) < need:
        raise SystemExit(f"합성 풀 부족: {len(pool)} < {need}. 템플릿 리스트를 늘리세요.")
    for t, cat, sub, sens in pool[:need]:
        if t not in seen:
            seen.add(t)
            rows.append([t, cat, sub, sens, "synthetic", 1])

    header = ["term", "category", "subcategory", "sensitivity", "source", "is_synthetic"]
    manifest = []
    for size in range(a.start, a.stop + 1, a.step):
        fn = out / f"dict_{size // 1000:03d}k.csv"
        with open(fn, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows[:size])
        manifest.append((fn.name, size, n_seed, size - n_seed))
        print(f"wrote {fn} ({size:,} terms, seed {n_seed}, synthetic {size - n_seed:,})")

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "terms", "seed_terms", "synthetic_terms"])
        w.writerows(manifest)


if __name__ == "__main__":
    main()
