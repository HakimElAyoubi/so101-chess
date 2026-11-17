"""Run every chess_sim test file and print a PASS/FAIL table.  From the project root:
    python chess_sim/tests/run_all.py [--only scene,env] [--skip motion] [--quiet]
Each file runs as a subprocess with this interpreter (its output is streamed unless --quiet); the per-test
PASS/FAIL lines and the final 'k/n passed' line are collected (35 tests: scene 13, motion 11, env 11).  The files write their own progress bars to
chess_sim/data/progress.txt (viewer: chess_sim/progress_view.command).  Full run 17-25 minutes (test_motion 14-22 min).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["test_scene.py", "test_motion.py", "test_env.py"]


def run_file(name: str, quiet: bool) -> dict:
    path = os.path.join(HERE, name)
    t0 = time.time()
    p = subprocess.Popen([sys.executable, path], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in p.stdout:
        if "objc[" in line:
            continue
        lines.append(line.rstrip("\n"))
        if not quiet:
            print(f"  | {line}", end="")
            sys.stdout.flush()
    p.wait()
    tests = [(m.group(1), m.group(2), float(m.group(3))) for l in lines for m in [re.match(r"^(PASS|FAIL) (\S+) \(([\d.]+)s\)", l)] if m]
    summ = [m for l in lines for m in [re.match(r"^(\d+)/(\d+) passed", l)] if m]
    n_pass, n_tot = (int(summ[-1].group(1)), int(summ[-1].group(2))) if summ else (0, 0)
    return dict(file=name, returncode=p.returncode, seconds=time.time() - t0, tests=tests, n_pass=n_pass, n_total=n_tot,
                tail=lines[-30:])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="comma list of suites: scene,motion,env")
    ap.add_argument("--skip", default=None, help="comma list of suites to skip")
    ap.add_argument("--quiet", action="store_true", help="do not stream the test output")
    a = ap.parse_args(argv)
    files = list(FILES)
    if a.only:
        keep = {s.strip() for s in a.only.split(",")}
        files = [f for f in files if f[len("test_"):-3] in keep]
    if a.skip:
        skip = {s.strip() for s in a.skip.split(",")}
        files = [f for f in files if f[len("test_"):-3] not in skip]
    t0 = time.time()
    results = []
    for f in files:
        print(f"=== {f} ===")
        sys.stdout.flush()
        results.append(run_file(f, a.quiet))
    total = time.time() - t0
    print("\n" + "=" * 78)
    print(f"{'suite':16s} {'test':38s} {'result':7s} {'time':>8s}")
    print("-" * 78)
    all_ok = True
    for r in results:
        for status, name, secs in r["tests"]:
            print(f"{r['file'][:-3]:16s} {name:38s} {status:7s} {secs:7.1f}s")
        ok = r["returncode"] == 0 and r["n_total"] > 0 and r["n_pass"] == r["n_total"]
        all_ok &= ok
        print(f"{r['file'][:-3]:16s} {'TOTAL %d/%d' % (r['n_pass'], r['n_total']):38s} {'PASS' if ok else 'FAIL':7s} {r['seconds']:7.1f}s")
        if not ok and r["n_total"] == 0:
            print("  (no summary line; last output lines:)")
            for l in r["tail"]:
                print("   ", l)
        print("-" * 78)
    print(f"{'ALL':16s} {'':38s} {'PASS' if all_ok else 'FAIL':7s} {total:7.1f}s  ({total / 60:.1f} min)")
    try:
        from chess_sim.config import PROGRESS_TXT
        with open(PROGRESS_TXT, "w") as fh:
            fh.write(f"[{'#' * 10}] 100%  {len(results)}/{len(results)}  elapsed {int(total) // 60}m{int(total) % 60:02d}s  ETA 0m00s | "
                     f"run_all: {'ALL PASS' if all_ok else 'FAILED'} {'DONE' if all_ok else 'FAILED'}\n")
    except Exception:
        pass
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    sys.exit(main())
