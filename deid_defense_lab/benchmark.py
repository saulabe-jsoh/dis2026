#!/usr/bin/env python3
"""사전 규모(1만~10만) × 저전력 프로파일 벤치마크.

측정: 사전 적재시간, 메모리(RSS), 스캔 처리량(자/초, CPU초 기준), 에너지(RAPL 가능 시),
      히트 수(시드/합성 구분), 주입(injection) 재현율, 스팬 덤프(JSONL).

예)  python benchmark.py --profile lowpower --samples samples --out results/lowpower.csv
     python benchmark.py --profile normal   --demo-corpus     --sizes 10000 50000 100000
"""
import argparse
import csv
import gc
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import unicodedata
from collections import deque
from datetime import datetime
from pathlib import Path

import psutil

try:
    import ahocorasick  # pip install pyahocorasick
    HAVE_AC = True
except ImportError:
    HAVE_AC = False


# ───────────── 매처 (pyahocorasick 우선, 없으면 순수 파이썬 폴백) ─────────────
class PyAC:
    """폴백용 Aho-Corasick. 느리지만 의존성 없음(폐쇄망 점검용)."""

    def __init__(self):
        self.goto, self.fail, self.val = [{}], [0], [None]

    def add_word(self, w, v):
        s = 0
        for ch in w:
            n = self.goto[s].get(ch)
            if n is None:
                self.goto.append({}); self.fail.append(0); self.val.append(None)
                n = len(self.goto) - 1
                self.goto[s][ch] = n
            s = n
        self.val[s] = v

    def make_automaton(self):
        q = deque(self.goto[0].values())
        while q:
            r = q.popleft()
            for ch, s in self.goto[r].items():
                q.append(s)
                f = self.fail[r]
                while f and ch not in self.goto[f]:
                    f = self.fail[f]
                t = self.goto[f].get(ch, 0)
                self.fail[s] = t if t != s else 0
                fv = self.val[self.fail[s]]
                if fv is not None and (self.val[s] is None or (fv >> 1) > (self.val[s] >> 1)):
                    self.val[s] = fv  # 같은 종료 위치에서 가장 긴 패턴 유지

    def iter(self, text):
        s = 0
        for i, ch in enumerate(text):
            while s and ch not in self.goto[s]:
                s = self.fail[s]
            s = self.goto[s].get(ch, 0)
            if self.val[s] is not None:
                yield i, self.val[s]


class Matcher:
    """값 = (길이<<1) | is_synthetic"""

    def __init__(self):
        self.a = ahocorasick.Automaton() if HAVE_AC else PyAC()

    def add(self, term, syn):
        self.a.add_word(term, (len(term) << 1) | syn)

    def finalize(self):
        self.a.make_automaton()

    def spans(self, text):
        """좌측우선·최장일치·비중첩 스팬 [(start,end,syn)]"""
        cand = []
        for end, v in self.a.iter(text):
            L = v >> 1
            cand.append((end + 1 - L, end + 1, v & 1))
        cand.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        out, last = [], -1
        for st, en, syn in cand:
            if st >= last:
                out.append((st, en, syn))
                last = en
        return out


# ───────────── 환경/프로파일 ─────────────
def apply_profile(a):
    prof = {"normal": dict(cores=0, nice=0, duty=100),
            "lowpower": dict(cores=1, nice=19, duty=50)}[a.profile]
    cores = a.cores if a.cores is not None else prof["cores"]
    nice = a.nice if a.nice is not None else prof["nice"]
    duty = a.duty if a.duty is not None else prof["duty"]
    p = psutil.Process()
    try:
        if cores > 0 and hasattr(p, "cpu_affinity"):
            allowed = p.cpu_affinity()
            p.cpu_affinity(allowed[:cores])
        if nice:
            p.nice(nice if os.name != "nt" else psutil.IDLE_PRIORITY_CLASS)
    except Exception as e:  # 권한 부족 등
        print(f"[warn] 프로파일 일부 적용 실패: {e}", file=sys.stderr)
    used = len(p.cpu_affinity()) if hasattr(p, "cpu_affinity") else psutil.cpu_count()
    return used, nice, duty


def cpu_model():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def read_rapl_uj():
    for p in Path("/sys/class/powercap").glob("intel-rapl:[0-9]"):
        try:
            return int((p / "energy_uj").read_text())
        except (OSError, ValueError):
            continue
    return None


# ───────────── 코퍼스 ─────────────
DEMO_SENTENCES = [
    "피고인은 2023. 3.경 서울 강남구 소재 회사 사무실에서 업무상 알게 된 자료를 저장매체에 복사하였다.",
    "위 기술자료는 회사의 영업비밀로서 비공지성, 경제적 유용성 및 비밀관리성을 갖추고 있다.",
    "피고인은 퇴사 직전 위 파일을 개인 이메일로 전송한 후 경쟁업체에 이직하였다.",
    "검사는 피고인에 대하여 징역 2년을 구형하였고, 변호인은 선처를 호소하였다.",
    "원심의 형은 재량의 합리적 범위를 벗어났다고 보기 어려워 항소를 기각한다.",
    "[이 사건 합성 데모 문장입니다. 실제 판결문이 아닙니다.]",
]


def load_corpus(a):
    docs = []
    d = Path(a.samples)
    if d.is_dir():
        for fp in sorted(d.glob("*.txt")):
            docs.append(unicodedata.normalize("NFC", fp.read_text(encoding="utf-8", errors="ignore")))
    if not docs:
        if not a.demo_corpus:
            sys.exit(f"{d}/ 에 .txt 판결문이 없습니다. fetch_judgments.py 실행 또는 --demo-corpus 사용")
        rng = random.Random(0)
        docs = [" ".join(rng.choices(DEMO_SENTENCES, k=60)) for _ in range(12)]
        print("[info] 합성 데모 코퍼스 사용 (성능 점검용, 실제 판결문 아님)")
    chunks = []
    for t in docs:
        chunks += [t[i:i + 2000] for i in range(0, len(t), 2000)]
    base = sum(map(len, chunks))
    reps = max(1, -(-a.target_chars // max(base, 1)))
    return chunks * reps, base


# ───────────── 단일 규모 실행 ─────────────
def run_one(a, size):
    cores, nice, duty = apply_profile(a)
    chunks, base_chars = load_corpus(a)
    total_chars = sum(map(len, chunks))
    proc = psutil.Process()

    path = Path(a.dict_dir) / f"dict_{size // 1000:03d}k.csv"
    gc.collect()
    rss0 = proc.memory_info().rss
    t0, c0 = time.perf_counter(), time.process_time()
    m = Matcher()
    pool_small = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            t = unicodedata.normalize("NFC", r["term"])
            m.add(t, int(r["is_synthetic"]))
            if i < 10_000:
                pool_small.append(t)  # 가장 작은 단계(모든 단계에 공통 포함)에서 주입용 샘플
    m.finalize()
    build_s = time.perf_counter() - t0
    build_cpu = time.process_time() - c0
    rss1 = proc.memory_info().rss

    # 스캔 (repeat회 중앙값)
    walls, cpus, actives, e_list = [], [], [], []
    hits_seed = hits_syn = 0
    dump = open(a.dump_spans, "w", encoding="utf-8") if a.dump_spans else None
    for rep in range(a.repeat):
        hs = hy = 0
        e0 = read_rapl_uj()
        w0, c0 = time.perf_counter(), time.process_time()
        active = 0.0
        for ci, ch in enumerate(chunks):
            s0 = time.perf_counter()
            sp = m.spans(ch)
            el = time.perf_counter() - s0
            active += el
            for st, en, syn in sp:
                if syn: hy += 1
                else: hs += 1
                if dump and rep == 0 and ci < base_chars // 2000 + 1:
                    dump.write(json.dumps({"chunk": ci, "start": st, "end": en, "text": ch[st:en],
                                           "synthetic": syn}, ensure_ascii=False) + "\n")
            if duty < 100:
                time.sleep(el * (100 - duty) / duty)
        walls.append(time.perf_counter() - w0)
        cpus.append(time.process_time() - c0)
        actives.append(active)
        e1 = read_rapl_uj()
        if e0 is not None and e1 is not None and e1 >= e0:
            e_list.append((e1 - e0) / 1e6)
        hits_seed, hits_syn = hs, hy
    if dump:
        dump.close()
    rss2 = proc.memory_info().rss

    # 주입 재현율: 공통 용어를 공백으로 감싸 삽입 → 정확 스팬 복원 여부
    rng = random.Random(7)
    n_inj = min(a.inject, len(chunks))
    ok = 0
    for ch in rng.sample(chunks, n_inj):
        term = rng.choice(pool_small)
        p = ch.find(" ", rng.randrange(0, max(1, len(ch) - 1)))
        p = len(ch) if p < 0 else p
        txt = ch[:p] + " " + term + " " + ch[p:]
        if any(st == p + 1 and en == p + 1 + len(term) for st, en, _ in m.spans(txt)):
            ok += 1

    med = statistics.median
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "profile": a.profile, "engine": "pyahocorasick" if HAVE_AC else "pure-python",
        "python": platform.python_version(), "cpu": cpu_model(),
        "cores_used": cores, "nice": nice, "duty_pct": duty,
        "dict_size": size, "build_s": round(build_s, 3), "build_cpu_s": round(build_cpu, 3),
        "dict_rss_mb": round((rss1 - rss0) / 2**20, 1), "rss_after_scan_mb": round(rss2 / 2**20, 1),
        "corpus_chars": total_chars, "scan_wall_s": round(med(walls), 3),
        "scan_cpu_s": round(med(cpus), 3), "scan_active_s": round(med(actives), 3),
        "chars_per_cpu_s": round(total_chars / max(med(cpus), 1e-9)),
        "avg_cpu_pct": round(100 * med(cpus) / max(med(walls), 1e-9), 1),
        "hits_seed": hits_seed, "hits_synthetic": hits_syn,
        "inject_n": n_inj, "inject_recall": round(ok / max(n_inj, 1), 4),
        "energy_j": round(med(e_list), 2) if e_list else "",
    }
    return row


def append_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["normal", "lowpower"], default="lowpower")
    ap.add_argument("--cores", type=int, help="사용 코어 수 (기본: normal=전체, lowpower=1)")
    ap.add_argument("--nice", type=int, help="우선순위 (lowpower 기본 19)")
    ap.add_argument("--duty", type=int, help="CPU 가동률 %% (lowpower 기본 50)")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(range(10_000, 100_001, 10_000)))
    ap.add_argument("--dict-dir", default="dict")
    ap.add_argument("--samples", default="samples")
    ap.add_argument("--demo-corpus", action="store_true")
    ap.add_argument("--target-chars", type=int, default=5_000_000, help="타이밍 안정화를 위해 코퍼스 반복")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--inject", type=int, default=500)
    ap.add_argument("--dump-spans", help="스팬 JSONL 경로(Thunder-DeID 결과와 병합용)")
    ap.add_argument("--out", default="results/results.csv")
    ap.add_argument("--no-isolate", action="store_true", help="규모별 서브프로세스 분리 안 함")
    a = ap.parse_args()

    if len(a.sizes) > 1 and not a.no_isolate:  # 규모마다 새 프로세스 → 메모리 측정 오염 방지
        for s in a.sizes:
            cmd = [sys.executable, __file__, "--profile", a.profile, "--sizes", str(s), "--no-isolate",
                   "--dict-dir", a.dict_dir, "--samples", a.samples, "--target-chars", str(a.target_chars),
                   "--repeat", str(a.repeat), "--inject", str(a.inject), "--out", a.out]
            for k in ("cores", "nice", "duty"):
                if getattr(a, k) is not None:
                    cmd += [f"--{k}", str(getattr(a, k))]
            if a.demo_corpus:
                cmd.append("--demo-corpus")
            if a.dump_spans:
                cmd += ["--dump-spans", a.dump_spans.replace(".jsonl", f"_{s // 1000:03d}k.jsonl")]
            subprocess.run(cmd, check=True)
        return

    for s in a.sizes:
        row = run_one(a, s)
        append_row(a.out, row)
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
