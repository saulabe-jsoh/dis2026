# 방산·안보수사 사전 × 판결문 비식별화 저전력 실험 키트

1. `pip install -r requirements.txt`  (Python 3.11 권장, 폐쇄망은 `pip download -r requirements.txt -d wheels` 후 반입)
2. `python generate_dict.py`  → `dict/dict_010k.csv … dict_100k.csv` (중첩 구조, 시드 60 + 합성)
3. `python fetch_judgments.py --n 15`  → `samples/*.txt` (LAW_OC 환경변수 필요, 인터넷 PC에서)
4. `python benchmark.py --profile lowpower --out results/lowpower.csv`
5. 결과 CSV 값을 `REPORT_방산사전_비식별화_실험보고서.md` 표에 옮겨 기입

저전력 프로파일 = 1코어 고정 + nice 19 + 가동률 50%. 실제 장비 전력은 OS 전원 계획/`cpulimit`/Docker `--cpus=0.5` 등으로 별도 제한 가능.
PowerShell 예: `$env:LAW_OC="id"; python benchmark.py --profile lowpower`

## v2 추가 (공개자료 기반 사전 · 판결문 후보)
- `dict/defense_public_dict_v1.csv` : 공개 법령·고시·판결 용어 324행 (출처ID·verify_status 포함)
- `dict/sources.csv`, `dict/judgment_candidates.csv` : 출처 목록 / 방산·산업기술 판결문 후보 13건
- `build_public_dict.py` (사전 재생성), `analyze_hits.py` (판결문별 히트 집계), `fetch_judgments.py --case-list dict/judgment_candidates.csv`
- `실험보고서_안보수사_방산사전_비식별화.docx` : 17쪽 보고서, 6장에 결과 입력란
