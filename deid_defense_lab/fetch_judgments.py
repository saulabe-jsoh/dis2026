#!/usr/bin/env python3
"""국가법령정보 공동활용(Open API) 판례 본문 수집기 → samples/*.txt

사전 준비 (1회):
  1) https://open.law.go.kr 회원가입 → OPEN API 신청(판례 목록/본문) → 승인
  2) 환경변수 LAW_OC 에 '이메일 ID'(@ 앞부분) 설정   예) export LAW_OC=myid
  3) 인터넷이 되는 PC에서 실행 후, samples/ 폴더를 수사망 PC로 옮김

사용:
  python fetch_judgments.py --n 15
  python fetch_judgments.py --n 20 --queries "군사기밀 보호법" "방위산업기술 보호법"

주의: 법원이 공개한 판례는 이미 비식별(○○○, A 등) 처리된 상태입니다.
     → 사전 히트·속도 실험에는 적합하나, '개인정보 탐지 성능'의 정답으로는 쓸 수 없습니다.
"""
import argparse
import csv
import html
import os
import re
import sys
import time
from pathlib import Path

import requests

BASE = "https://www.law.go.kr/DRF"
DEFAULT_QUERIES = ["군사기밀 보호법", "방위산업기술 보호법", "산업기술의 유출방지", "방위사업법 위반",
                   "영업비밀 국외누설", "군사기밀 누설", "전략물자 수출", "부정경쟁방지 기술유출"]
TEXT_KEYS = ["법원명", "사건명", "사건번호", "선고일자", "판시사항", "판결요지", "참조조문", "참조판례", "판례내용"]


def find_key(obj, key):
    """중첩 JSON에서 key 재귀 탐색 (응답 구조 변화에 대비)"""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def clean(s):
    s = html.unescape(str(s or ""))
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def search(oc, query, page, display=20):
    r = requests.get(f"{BASE}/lawSearch.do", timeout=30,
                     params=dict(OC=oc, target="prec", type="JSON", query=query, display=display, page=page))
    r.raise_for_status()
    items = find_key(r.json(), "prec") or []
    return items if isinstance(items, list) else [items]


def service(oc, pid):
    r = requests.get(f"{BASE}/lawService.do", timeout=30, params=dict(OC=oc, target="prec", ID=pid, type="JSON"))
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="저장할 건수 (10건 이상 권장)")
    ap.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    ap.add_argument("--out", default="samples")
    ap.add_argument("--min-chars", type=int, default=800, help="너무 짧은 본문 제외")
    ap.add_argument("--case-list", help="dict/judgment_candidates.csv: 사건번호로 직접 검색(국가법령정보 DB에 있는 건만 저장)")
    a = ap.parse_args()

    oc = os.environ.get("LAW_OC")
    if not oc:
        sys.exit("환경변수 LAW_OC 가 필요합니다 (open.law.go.kr 가입 ID).")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seen, rows = set(), []
    if a.case_list:
        import csv as _csv
        for r in _csv.DictReader(open(a.case_list, encoding="utf-8-sig")):
            no = r["case_no"].split()[0].replace("외", "").strip()
            try:
                rr = requests.get(f"{BASE}/lawSearch.do", timeout=30, params=dict(
                    OC=oc, target="prec", type="JSON", nb=no, display=5))  # nb=사건번호
                items = find_key(rr.json(), "prec") or []
                items = items if isinstance(items, list) else [items]
                hit = [i for i in items if no in str(i.get("사건번호", ""))]
            except Exception as e:
                print(f"[warn] {no}: {e}")
                continue
            if not hit:
                print(f"[miss] {no} — 국가법령정보 DB에 없음 → LBOX/종합법률정보에서 수동 저장")
                continue
            pid = str(hit[0].get("판례일련번호", "")).strip()
            if pid in seen:
                continue
            seen.add(pid)
            data = service(oc, pid)
            text = "\n\n".join(f"[{k}]\n{clean(find_key(data, k))}" for k in TEXT_KEYS if clean(find_key(data, k)))
            (out / f"prec_{pid}.txt").write_text(text, encoding="utf-8")
            rows.append([pid, no, clean(find_key(data, "사건명")), clean(find_key(data, "선고일자")), len(text), "case-list"])
            print(f"saved prec_{pid}.txt {no}")
            time.sleep(0.5)
    for q in (a.queries if len(rows) < a.n else []):
        for page in (1, 2):
            if len(rows) >= a.n:
                break
            try:
                items = search(oc, q, page)
            except Exception as e:
                print(f"[warn] 검색 실패 {q} p{page}: {e}")
                continue
            for it in items:
                pid = str(it.get("판례일련번호", "")).strip()
                if not pid or pid in seen or len(rows) >= a.n:
                    continue
                seen.add(pid)
                try:
                    data = service(oc, pid)
                except Exception as e:
                    print(f"[warn] 본문 실패 {pid}: {e}")
                    continue
                parts = []
                for k in TEXT_KEYS:
                    v = clean(find_key(data, k))
                    if v:
                        parts.append(f"[{k}]\n{v}")
                text = "\n\n".join(parts)
                if len(text) < a.min_chars:
                    continue
                (out / f"prec_{pid}.txt").write_text(text, encoding="utf-8")
                rows.append([pid, clean(find_key(data, "사건번호")), clean(find_key(data, "사건명")),
                             clean(find_key(data, "선고일자")), len(text), q])
                print(f"saved prec_{pid}.txt ({len(text):,}자) {rows[-1][1]} {rows[-1][2]}")
                time.sleep(0.5)  # 서버 예의
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "case_no", "case_name", "date", "chars", "query"])
        w.writerows(rows)
    print(f"\n완료: {len(rows)}건 → {out}/  (목표 {a.n})")
    if len(rows) < 10:
        print("10건 미만입니다. --queries 를 늘리거나 법원 종합법률정보(glaw.scourt.go.kr)에서 수동 저장하세요.")


if __name__ == "__main__":
    main()
