#!/usr/bin/env bash
# 전체 실험 순서 (Linux). Windows는 README의 PowerShell 예시 참고.
set -e
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python generate_dict.py --out dict                 # 1만~10만 사전 생성
# python fetch_judgments.py --n 15                  # (인터넷 PC) 판결문 수집 → samples/
python benchmark.py --profile normal   --samples samples --out results/normal.csv   --dump-spans results/spans.jsonl
python benchmark.py --profile lowpower --samples samples --out results/lowpower.csv
