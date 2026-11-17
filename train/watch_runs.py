"""Live progress of the step-3 training runs on the rented GPU boxes, for data/progress.html (auto-refreshing).

Every 60 s, over SSH: the training log's step count (one "loss:" line per --log_freq=100 steps), the last loss, and
whether the run finished, crashed or the watcher (after_training.sh) uploaded everything. One bar per run, with
percentage, step, elapsed time and ETA, written to data/progress.txt.

    python train/watch_runs.py smolvla_skills=ssh4.vast.ai:12345:20000 smolvla_skills_vision=...:20000 act_skills=...:40000
"""
import re, subprocess, sys, time
from pathlib import Path

PROG = Path(__file__).resolve().parent.parent / "data/progress.txt"
LOG_FREQ = 100
KEY = Path.home() / ".ssh/vast_softgroup"         # the key registered on the Vast.ai account ("rag-pilot-vast")


def probe(host, port, run):
    log = f"~/train_{run}.log"                       # tagged lines: missing ones don't shift the others
    cmd = (f"echo N=$(grep -c 'loss:' {log} 2>/dev/null); echo L=$(grep -o ' loss:[0-9.]*' {log} 2>/dev/null | tail -1); "
           f"echo E=$(grep -E -c 'Traceback|CUDA out of memory|RuntimeError' {log} 2>/dev/null); "
           f"echo A=$(tail -1 ~/after_training.log 2>/dev/null); "
           f"pgrep -f '[v]env/bin/lerobot-train' >/dev/null && echo S=RUNNING || echo S=STOPPED")   # [v]: don't match this shell
    r = subprocess.run(["ssh", "-i", str(KEY), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                        "-o", "StrictHostKeyChecking=accept-new", "-p", port, f"root@{host}", cmd],
                       capture_output=True, text=True, timeout=40)
    return r.stdout.strip().splitlines()


def main():
    runs = []
    for a in sys.argv[1:]:
        name, spec = a.split("=", 1)
        host, port, steps = spec.split(":")
        runs.append(dict(name=name, host=host, port=port, steps=int(steps), t0=None, state="starting", step=0, loss="", hist=[]))
    T0 = time.time()
    while True:
        for r in runs:
            try:
                out = dict(l.split("=", 1) for l in probe(r["host"], r["port"], r["name"]) if "=" in l)
                if "S" not in out:
                    raise ConnectionError("no answer")
                r["step"] = int(out.get("N") or 0) * LOG_FREQ
                m = re.search(r"loss:([0-9.]+)", out.get("L", ""))
                r["loss"] = m.group(1) if m else r["loss"]
                errors = int(out.get("E") or 0)
                running = out["S"] == "RUNNING"
                after = out.get("A", "")
                r["hist"] = [h for h in r["hist"] + [(time.time(), r["step"])] if h[0] > time.time() - 600]
                r["state"] = ("uploaded, machine stopping" if "everything is on the Hub" in after else
                              "ERROR (see the log)" if errors and not running else
                              "training" if running else "finished" if r["step"] >= r["steps"] else "setting up")
            except Exception as e:
                r["state"] = f"unreachable ({type(e).__name__})"
        lines = []
        for r in runs:
            f = min(1.0, r["step"] / r["steps"])
            (t1, s1), (t2, s2) = r["hist"][0], r["hist"][-1]       # rate over the last 10 min of polls
            rate = (s2 - s1) / (t2 - t1) if t2 - t1 > 90 and s2 > s1 else 0
            el = r["step"] / rate if rate else 0                    # time spent training so far, at that rate
            eta = (r["steps"] - r["step"]) / rate if rate else 0
            done_at = time.strftime("%H:%M", time.localtime(time.time() + eta)) if rate else "--:--"
            lines.append(f"[{'#' * int(30 * f):<30}] {100 * f:5.1f}%  {r['step']}/{r['steps']} steps  {rate:4.2f} steps/s  "
                         f"loss {r['loss'] or '-'}  elapsed {el / 60:4.0f} min  ETA {eta / 60:4.0f} min (~{done_at}) | {r['name']}: {r['state']}")
        done = PROG.with_name("progress_done.txt")         # runs already finished (kept on the page)
        head = done.read_text() if done.exists() else ""
        PROG.write_text(head + "\n".join(lines) + f"\n(updated {time.strftime('%H:%M:%S')}, watching for {(time.time() - T0) / 60:.0f} min)\n")
        if all(r["state"].startswith(("uploaded", "ERROR")) for r in runs):
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
