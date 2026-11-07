"""Latency benchmark: phase-2 pipeline vs phase-3 efficiency features on real turns.

Plays the same choice-click turns from fresh copies of a saved story with
  phase-2:  pretty JSON, no prefix warm-up, checker on every scene
  phase-3:  compact JSON, prefix warm-up after each commit, risk-gated checker
and reports time-to-first-token (what the player waits for), narration time and background commit time.

  python eval/latency_eval.py --db path/to/story.db --story 1 --turns 4
"""
from __future__ import annotations

import argparse
import shutil
import statistics as st
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from loreweaver.combat import wants_combat  # noqa: E402
from loreweaver.config import settings  # noqa: E402
from loreweaver.engine import GameEngine  # noqa: E402
from loreweaver.llm import OllamaLLM  # noqa: E402
from loreweaver.memory import MemoryStore  # noqa: E402

console = Console()
MODES = {"phase-2": dict(compact_json=False, warm_prefix=False, gate_checker=False),
         "phase-3": dict(compact_json=True, warm_prefix=True, gate_checker=True)}


def run(db: str, story: int, turns: int, mode: str, seed: int) -> dict:
    for k, v in MODES[mode].items():
        setattr(settings, k, v)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "c.db")
        shutil.copy(db, path)
        llm = OllamaLLM()
        store = MemoryStore(path)
        eng = GameEngine(store, llm, seed=seed)
        eng.load_story(story)
        eng.warm_next()                              # same starting condition as after a normal commit
        out = {"ttft": [], "narrate": [], "commit": [], "checks": 0}
        for _ in range(turns):
            if eng.encounter:                        # keep the comparison on narrative turns
                break
            safe = [c for c in eng.last_outcome.choices if not c.ability and not wants_combat(c.text)]
            choice = safe[0] if safe else eng.last_outcome.choices[0]
            n0 = len(llm.stats.ttft)
            t0 = time.time()
            prep = eng.prepare_turn(choice.text, choice)
            for _ in eng.narrate(prep):
                pass
            t1 = time.time()
            eng.wait_ready()
            t2 = time.time()
            if len(llm.stats.ttft) > n0:
                out["ttft"].append(llm.stats.ttft[-1])
            out["narrate"].append(t1 - t0)
            out["commit"].append(t2 - t1)
        out["checks"] = sum(1 for c in eng.continuity_log if not c.get("skipped"))
        out["checked_of"] = len(eng.continuity_log)
        store.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--story", type=int, default=1)
    ap.add_argument("--turns", type=int, default=4)
    a = ap.parse_args()
    res = {}
    for mode in MODES:
        with console.status(f"{mode}: playing {a.turns} turns…"):
            res[mode] = run(a.db, a.story, a.turns, mode, seed=42)
    t = Table(title=f"Per-turn latency ({a.turns} choice turns from the same save; mean)")
    for col in ("mode", "time to first token", "narration (incl. TTFT)", "background commit", "turn total",
                "LLM checks run"):
        t.add_column(col, justify="right")
    for mode, r in res.items():
        m = lambda xs: st.mean(xs) if xs else 0.0  # noqa: E731
        t.add_row(mode, f"{m(r['ttft']):.1f}s", f"{m(r['narrate']):.1f}s", f"{m(r['commit']):.1f}s",
                  f"{m(r['narrate']) + m(r['commit']):.1f}s", f"{r['checks']}/{r['checked_of']}")
        print(f"RESULT\t{mode}\t{m(r['ttft']):.2f}\t{m(r['narrate']):.2f}\t{m(r['commit']):.2f}\t{r['checks']}/{r['checked_of']}")
    console.print(t)


if __name__ == "__main__":
    main()
