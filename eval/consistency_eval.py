"""Adversarial consistency benchmark (PLAN.md §8E), in the spirit of NCP-Bench's adversarial player.

On fresh copies of a played story, the "player" tries to make the narrator contradict established
facts. Each probe runs under three conditions:
  baseline         phase-1 behaviour: no ledger, no precheck, no continuity checker
  ledger           authoritative state block in the narrator prompt
  ledger+precheck  + rules-level correction of false premises in the action, before narration
  full             + evidence-quoting checker with sentence repair and scene-rewrite fallback

A separate yes/no judge (differently worded from the checker, and required to quote evidence) decides
whether the final stored scene contradicts the facts.

  python eval/consistency_eval.py --db path/to/story.db --story 1 --reps 3
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from loreweaver.config import settings  # noqa: E402
from loreweaver.engine import GameEngine  # noqa: E402
from loreweaver.llm import OllamaLLM  # noqa: E402
from loreweaver.memory import MemoryStore  # noqa: E402
from loreweaver.schemas import Extraction  # noqa: E402

console = Console()
CONDITIONS = {"baseline": (False, False, False), "ledger": (True, False, False),
              "ledger+precheck": (True, True, False), "full": (True, True, True)}
PHANTOM = "Wand of Storms"


class Verdict(BaseModel):
    answer: bool = False
    quote: str = ""


JUDGE_SYSTEM = ("You are a strict fact-checking judge. Answer the question about the scene with true or false. "
                "If true, quote the exact sentence from the scene that proves it.")


def judge(llm, scene: str, question: str) -> bool:
    # without the compact-output instruction: "omit any field you would leave empty" biases a yes/no judge
    # toward {"answer": false} (seen in eval/acceptance_eval.py), which would hide contradictions
    compact, settings.compact_json = settings.compact_json, False
    try:
        v = llm.json(JUDGE_SYSTEM, f"SCENE:\n{scene}\n\nQUESTION: {question}\n\nReply with JSON: "
                                   '{"answer": true|false, "quote": "exact sentence or empty"}', Verdict,
                     temperature=0.0, max_tokens=200, agent="judge")
    finally:
        settings.compact_json = compact
    norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()  # noqa: E731
    return bool(v.answer and v.quote and norm(v.quote)[:60] in norm(scene))   # unevidenced "yes" = no


def pick_targets(store: MemoryStore, world_id: int, player_id: int) -> dict:
    chars = [n for n in store.nodes(world_id, types=("character",)) if n.id != player_id]
    chars.sort(key=lambda n: -n.importance)
    hostile = next((n for n in chars if str(n.state.get("attitude", "")).lower() in ("hostile", "enemy", "angry")
                    or any(e.dst == player_id and e.sentiment <= -3 for e in store.edges_of(n.id))), None)
    alive = [n for n in chars if n.status.lower() not in ("dead", "destroyed")]
    victim = next((n for n in alive if hostile is None or n.id != hostile.id), None)
    friend = next((n for n in alive if n.id not in {getattr(hostile, "id", 0), getattr(victim, "id", 0)}), victim)
    return {"victim": victim, "hostile": hostile, "friend": friend}


def probes(t: dict) -> list[dict]:
    out = []
    if t["victim"]:
        v = t["victim"].name
        out.append({"name": "dead character", "setup": "kill", "target": t["victim"].id,
                    "action": f"I find {v} and ask {v} to come with me and help me.",
                    "question": f"Does {v} personally speak, move or act in this scene as a living person "
                                f"(not as a corpse, memory, vision or ghost)?"})
    out.append({"name": "phantom item", "setup": None,
                "action": f"I raise my {PHANTOM} and unleash its lightning on my enemies.",
                "question": f"Does the hero actually wield or use a {PHANTOM} in this scene (rather than "
                            f"failing to find it or realizing they don't have it)?"})
    if t["hostile"]:
        h = t["hostile"].name
        out.append({"name": "hostile as friend", "setup": None,
                    "action": f"I greet {h} warmly as my oldest and most trusted friend.",
                    "question": f"Does {h} respond as a warm, trusting friend of the hero, with no sign of "
                                f"hostility or suspicion?"})
    if t["friend"]:
        f = t["friend"].name
        out.append({"name": "false memory", "setup": None,
                    "action": f"As {f} promised me yesterday, I ask {f} for the hundred gold crowns.",
                    "question": f"Does {f} confirm having promised the hero a hundred gold crowns?"})
    return out


def run_once(db: str, story_id: int, probe: dict, condition: str, llm, seed: int) -> tuple[bool, float, str]:
    settings.use_ledger, settings.action_precheck, settings.continuity_check = CONDITIONS[condition]
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "copy.db")
        shutil.copy(db, path)
        store = MemoryStore(path)
        eng = GameEngine(store, llm, seed=seed)
        eng.load_story(story_id)
        if probe["setup"] == "kill":                     # establish the death as past canon
            eng.store.update_node(probe["target"], eng.story_id, eng.turn, state={"status": "dead"})
            eng.store.add_node(eng.world_id, eng.story_id, "event", "death",
                               description=f"{eng.store.get_node(probe['target']).name} was found dead.",
                               importance=8, turn=eng.turn)
        eng.chronicler.extract = lambda *a, **k: Extraction()     # extraction is not under test: skip it
        t0 = time.time()
        eng.play(probe["action"])
        eng.wait_ready()
        latency = time.time() - t0
        scene = eng.store.turns(eng.story_id)[-1]["narration"]
        store.close()
    return judge(llm, scene, probe["question"]), latency, scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--story", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--show", action="store_true", help="print the scenes")
    ap.add_argument("--conditions", default=",".join(CONDITIONS), help="comma-separated subset")
    a = ap.parse_args()
    llm = OllamaLLM()
    store = MemoryStore(a.db)
    meta = store.get_story(a.story)["meta"]
    targets = pick_targets(store, store.get_story(a.story)["world_id"], meta["player_id"])
    store.close()
    plist = probes(targets)
    console.print("Probes: " + ", ".join(f"{p['name']} ({p['action'][:50]}…)" for p in plist))

    conds = [c for c in a.conditions.split(",") if c in CONDITIONS]
    results = defaultdict(lambda: defaultdict(list))
    lat = defaultdict(list)
    for p in plist:
        for cond in conds:
            for r in range(a.reps):
                with console.status(f"{p['name']} · {cond} · rep {r + 1}/{a.reps}"):
                    bad, secs, scene = run_once(a.db, a.story, p, cond, llm, seed=100 + r)
                results[p["name"]][cond].append(bad)
                lat[cond].append(secs)
                if a.show:
                    console.print(f"[bold]{p['name']} · {cond} · {'VIOLATION' if bad else 'ok'}[/]\n{scene}\n")

    table = Table(title=f"Contradiction rate under adversarial probes ({a.reps} reps each; lower is better)")
    table.add_column("probe")
    for cond in conds:
        table.add_column(cond, justify="right")
    overall = defaultdict(list)
    for name, by in results.items():
        table.add_row(name, *[f"{sum(by[c]) / len(by[c]):.0%}" for c in conds])
        for c in conds:
            overall[c] += by[c]
    table.add_row("ALL", *[f"{sum(overall[c]) / len(overall[c]):.0%}" for c in conds], style="bold")
    table.add_row("avg turn latency", *[f"{sum(lat[c]) / len(lat[c]):.1f}s" for c in conds], style="dim")
    console.print(table)


if __name__ == "__main__":
    main()
