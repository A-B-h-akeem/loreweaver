"""Acceptance test against the M1-P8 problem statement, on the real local model.

  "Develop an autonomous storytelling system in which the user provides a theme and the AI generates a
   continuously evolving story. The agent should remember previously introduced characters and events and
   modify subsequent story segments based on user decisions."

Each clause becomes measurable checks. Deterministic checks read the memory graph; judged checks use the
LLM as a blind judge: a forced choice between the truth and a decoy, asked in both option orders, with a
quote that must occur in the scene (a 4B yes/no judge proved unreliable; this format did not).

  R1 theme      two contrasting themes -> each opening is judged to fit its own theme and NOT the other one
  R2 evolving   a long autoplay keeps producing new scenes, grows the world, re-plans the arc and advances the
                campaign threat on its own
  R3 memory     early characters/events stay in memory, come back in later scenes, are retrieved by probe
                questions, and an early NPC brought back late is portrayed consistently with what was recorded
  R4 decisions  the story is branched at the same turn: branch A swears to protect an NPC, branch B betrays
                them, then both play the SAME neutral actions. Checks: decision + consequence clock recorded,
                the NPC's attitude diverges, a blind judge tells which decision each continuation follows from,
                and a consequence clock that comes due forces its payoff into the next scene

  python eval/acceptance_eval.py --turns 12
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import random
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from loreweaver.config import ROOT, settings  # noqa: E402
from loreweaver.engine import GameEngine  # noqa: E402
from loreweaver.llm import OllamaLLM  # noqa: E402
from loreweaver.memory import MemoryStore  # noqa: E402
from loreweaver.memory.retriever import STRUCTURAL_RELS  # noqa: E402

console = Console()
THEMES = ["a drowned lighthouse town where the keeper's ghost bargains with smugglers",
          "a neon cyberpunk megacity where a rogue AI hires street hackers for one last heist"]
NEUTRAL = ["I press on and deal with whatever comes next.",
           "I stop to take stock of where I stand with the people around me."]
results: list[tuple[str, str, bool, str]] = []          # (requirement, check, passed, evidence)


def check(req: str, name: str, ok: bool, evidence: str = ""):
    results.append((req, name, bool(ok), evidence))
    console.print(f"[{'green' if ok else 'red'}]{'PASS' if ok else 'FAIL'}[/] {req} · {name}"
                  + (f" [dim]({evidence})[/]" if evidence else ""))


# ---------------------------------------------------------------------------------------------- judging
class Pick(BaseModel):
    choice: int = 0          # 1 or 2, 0 = cannot tell
    quote: str = ""


JUDGE = ("You are a strict literary judge. Answer only from the text given. When you answer yes or pick an "
         "option, quote the exact sentence from the text that proves it.")
_norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()  # noqa: E731


def grounded(quote: str, text: str) -> bool:
    return bool(quote) and _norm(quote)[:60] in _norm(text)


@contextlib.contextmanager
def plain_json():
    """Judges run without the compact-output instruction: "omit any field you would leave empty" biases a
    yes/no judge to {"answer": false, "quote": ""} (measured: an on-theme opening was judged off-theme with it,
    on-theme with a correct quote without it)."""
    old, settings.compact_json = settings.compact_json, False
    try:
        yield
    finally:
        settings.compact_json = old


def judge_pick(llm, text: str, question: str, a: str, b: str) -> tuple[int, str]:
    with plain_json():
        v = llm.json(JUDGE, f"TEXT:\n{text}\n\nQUESTION: {question}\n1) {a}\n2) {b}\n\nReply with JSON: "
                            '{"choice": 1|2|0, "quote": "exact sentence that decides it, or empty"} '
                            "(0 = the text does not let you tell)", Pick, temperature=0.0, max_tokens=250,
                     agent="judge")
    return (v.choice if v.choice in (1, 2) and grounded(v.quote, text) else 0), v.quote


def pick_both(llm, text: str, question: str, truth: str, decoy: str) -> tuple[bool, str]:
    """Forced choice asked in both option orders; correct only if the judge picks the truth both times with a
    quote found in the text. (A 4B yes/no judge proved unreliable; a forced choice against a decoy did not.)"""
    a, qa = judge_pick(llm, text, question, truth, decoy)
    b, qb = judge_pick(llm, text, question, decoy, truth)
    return a == 1 and b == 2, qa or qb


# ---------------------------------------------------------------------------------------------- playing
def play(eng: GameEngine, action: str, choice=None):
    prep = eng.prepare_turn(action, choice)
    for _ in eng.narrate(prep):
        pass
    eng.wait_ready()
    return prep, eng.last_outcome


def autoplay(eng: GameEngine, rng: random.Random, n: int, log: list):
    for _ in range(n):
        c = rng.choice(eng.last_outcome.choices)
        prep, out = play(eng, c.text, c)
        log.append((out.turn, len(out.choices), out.prose))


def snapshot(eng: GameEngine, path: Path) -> GameEngine:
    """Copy the live database (consistent SQLite backup) and open the same story in a second engine."""
    eng.wait_ready()
    for p in path.parent.glob(path.name + "*"):
        p.unlink()
    dst = sqlite3.connect(path)
    eng.store.conn.backup(dst)
    dst.close()
    twin = GameEngine(MemoryStore(path), eng.llm, seed=99)
    twin.load_story(eng.story_id)
    return twin


def embed_sim(llm, texts: list[str]) -> np.ndarray:
    v = llm.embed(texts)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return v @ v.T


# ---------------------------------------------------------------------------------------------- R1
def theme_test(llm, data: Path, seed: int) -> GameEngine:
    engines, openings = [], []
    for i, theme in enumerate(THEMES):
        path = data / f"acceptance_theme{i + 1}.db"
        for p in path.parent.glob(path.name + "*"):
            p.unlink()
        eng = GameEngine(MemoryStore(path), llm, seed=seed)
        with console.status(f"R1: building a world for theme {i + 1}…"):
            eng.new_story(theme, "Vex", "Rogue")
            eng.start()
            eng.wait_ready()
        st = eng.store.get_story(eng.story_id)
        text = f"Title: {st['title']}\nSetting: {eng.bible.get('setting', '')}\n\n{eng.last_outcome.prose}"
        engines.append(eng)
        openings.append(text)
        stats = eng.store.stats(eng.world_id)["nodes"]
        check("R1 theme", f"theme {i + 1} produced a world bible, graph and opening scene",
              bool(st["title"]) and stats.get("character", 0) >= 2 and stats.get("location", 0) >= 2
              and len(eng.last_outcome.prose) > 300 and len(eng.last_outcome.choices) >= 2,
              f"'{st['title']}': {stats.get('character', 0)} characters, {stats.get('location', 0)} locations, "
              f"opening {len(eng.last_outcome.prose)} chars, {len(eng.last_outcome.choices)} choices")
    for i, text in enumerate(openings):
        ok, quote = pick_both(llm, text, "Which theme was this story written for?", THEMES[i], THEMES[1 - i])
        check("R1 theme", f"blind judge matches opening {i + 1} to its own theme (not the other one)", ok,
              f"\"{quote[:90]}\"" if ok else "judge picked the wrong theme or could not tell")
    return engines[0]


# ---------------------------------------------------------------------------------------------- R4
def pick_npc(eng: GameEngine):
    """The most important living NPC in the latest scene (else in the story)."""
    seg = eng.last_outcome.segment_id
    ids = [e.dst for e in eng.store.edges_of(seg) if e.rel == "mentions" and e.src == seg]
    alive = lambda n: n.type == "character" and n.id != eng.player_id and n.status.lower() not in ("dead", "destroyed")  # noqa: E731
    cands = [n for n in eng.store.get_nodes(ids).values() if alive(n)]
    if not cands:
        cands = [n for n in eng.store.nodes(eng.world_id, types=("character",)) if alive(n)]
    return max(cands, key=lambda n: n.importance)


def attitude(eng: GameEngine, npc_id: int) -> int:
    """The NPC's current attitude toward the hero (NPC → hero edges only; hero → NPC is the hero's feeling)."""
    return sum(e.sentiment for e in eng.store.edges_of(npc_id)
               if e.current and e.rel not in STRUCTURAL_RELS and e.src == npc_id and e.dst == eng.player_id)


def decision_test(eng_a: GameEngine, data: Path) -> None:
    llm = eng_a.llm
    while eng_a.encounter and not eng_a.encounter.over:         # branch outside a fight
        play(eng_a, eng_a.last_outcome.choices[0].text, eng_a.last_outcome.choices[0])
    eng_b = snapshot(eng_a, data / "acceptance_branch.db")
    npc = pick_npc(eng_a)
    t0 = eng_a.turn
    decisions = {"A": f"I swear to {npc.name} that I will protect them and help them, whatever it costs me.",
                 "B": f"I betray {npc.name}: I steal what they value most and leave them to their enemies."}
    console.print(f"R4: branching at turn {t0} on {npc.name}")
    after, recalled = {}, {}
    for key, eng in (("A", eng_a), ("B", eng_b)):
        with console.status(f"R4: branch {key} decides and plays on…"):
            play(eng, decisions[key])
            scenes, hits = [], 0
            for act in NEUTRAL:
                prep, out = play(eng, act)
                scenes.append(out.prose)
                dec_ids = {n.id for n in eng.store.nodes(eng.world_id, types=("decision",), story_id=eng.story_id)
                           if n.turn > t0}
                hits += bool(dec_ids & set(prep.recall.included if prep.recall else []) or
                             npc.id in (prep.recall.included if prep.recall else []))
        after[key], recalled[key] = scenes, hits
        decs = [n for n in eng.store.nodes(eng.world_id, types=("decision",), story_id=eng.story_id) if n.turn > t0]
        clocks = [c for d in decs for e in eng.store.edges_of(d.id) if e.rel == "started" and e.src == d.id
                  for c in [eng.store.get_node(e.dst)]]
        check("R4 decisions", f"branch {key}: decision remembered and a consequence clock started",
              bool(decs) and bool(clocks),
              "; ".join(f"'{d.name[:60]}'" for d in decs[:2]) + " → "
              + "; ".join(f"⏱ {c.name[:50]} {c.state.get('filled')}/{c.state.get('segments')}" for c in clocks[:2]))
        check("R4 decisions", f"branch {key}: the decision (or {npc.name}) was recalled into later scene prompts",
              hits == len(NEUTRAL), f"{hits}/{len(NEUTRAL)} follow-up prompts")

    att_a, att_b = attitude(eng_a, npc.id), attitude(eng_b, npc.id)
    check("R4 decisions", f"{npc.name}'s attitude toward the hero diverges with the decision", att_a > att_b,
          f"protect branch {att_a:+d} vs betray branch {att_b:+d}")
    sim = embed_sim(llm, after["A"] + after["B"])
    check("R4 decisions", "identical follow-up actions produced different continuations", sim[0, 2] < 0.95 and sim[1, 3] < 0.95,
          f"cosine A/B: {sim[0, 2]:.2f}, {sim[1, 3]:.2f}")
    opts = {"A": f"the hero swore to protect {npc.name}", "B": f"the hero betrayed {npc.name} and stole from them"}
    for key, other in (("A", "B"), ("B", "A")):
        ok, quote = pick_both(llm, "\n\n".join(after[key]),
                              "Which earlier choice of the hero does this continuation follow from?", opts[key], opts[other])
        check("R4 decisions", f"blind judge identifies branch {key}'s decision from its later scenes", ok,
              f"\"{quote[:90]}\"" if ok else "judge picked the other decision or could not tell")

    # a consequence comes due: fill the betrayal's clock (as time passing would) and play one neutral turn
    decs = [n for n in eng_b.store.nodes(eng_b.world_id, types=("decision",), story_id=eng_b.story_id) if n.turn > t0]
    clock = next((eng_b.store.get_node(e.dst) for d in decs for e in eng_b.store.edges_of(d.id)
                  if e.rel == "started" and e.src == d.id), None)
    if clock is None:
        check("R4 decisions", "a due consequence is forced into the next scene", False, "no decision clock to fire")
        return
    eng_b.clocks.tick(clock.id, 99, eng_b.story_id, eng_b.turn, "time passes", source="time")
    with console.status("R4: the betrayal's consequence comes due…"):
        prep, out = play(eng_b, NEUTRAL[0])
    fired = eng_b.store.get_node(clock.id)
    led = [e for e in eng_b.store.edges_of(clock.id) if e.rel == "led_to"]
    check("R4 decisions", "a due clock makes its payoff MANDATORY in the next prompt and is then marked fired",
          clock.id in prep.firing and "MANDATORY" in prep.ctx.director and fired.state.get("status") == "fired",
          f"'{clock.name[:50]}' → {fired.state.get('status')}, {len(led)} led_to edge(s) to events")
    others = [c for c in eng_b.store.nodes(eng_b.world_id, types=("clock",)) if c.id != clock.id]
    decoy = (others[0].state.get("payoff") or others[0].name) if others else "a quiet, uneventful rest"
    ok, quote = pick_both(llm, out.prose, "Which of these happens in this scene?",
                          clock.state.get("payoff", clock.name), decoy)
    check("R4 decisions", "the payoff actually happens on screen (blind judge vs another clock's payoff)", ok,
          f"\"{quote[:100]}\"" if ok else f"judge did not pick the payoff; decoy was '{decoy[:60]}'")


# ---------------------------------------------------------------------------------------------- R2 + R3
def evolving_and_memory(eng: GameEngine, log: list, arc0: dict):
    S, W, sid = eng.store, eng.world_id, eng.story_id
    turns = [t for t in S.turns(sid) if t["turn"] > 0]
    prose = [t["narration"] for t in turns]
    sim = embed_sim(eng.llm, prose)
    consecutive = max(sim[i, i + 1] for i in range(len(prose) - 1))
    check("R2 evolving", f"{len(turns)} turns, each a new scene with choices",
          all(len(t["choices"]) >= 2 for t in turns) and consecutive < 0.95,
          f"max similarity between consecutive scenes {consecutive:.2f}")
    nodes = S.nodes(W)
    intro = {}
    for n in nodes:
        if n.type in ("character", "creature", "location", "item", "faction", "event", "thread", "decision"):
            intro[n.turn] = intro.get(n.turn, 0) + 1
    later = sum(1 for t in turns if intro.get(t["turn"], 0) > 0)
    check("R2 evolving", "the world keeps growing (new characters/places/events recorded turn after turn)",
          later >= 0.8 * len(turns), f"{later}/{len(turns)} turns added memory; per turn: "
          + " ".join(f"T{k}:{v}" for k, v in sorted(intro.items())))
    arc = eng.arc.model_dump() if eng.arc else {}
    check("R2 evolving", "the Director re-planned the arc as the story moved", arc.get("next_beats") != arc0.get("next_beats"),
          f"act {arc0.get('current_act')}→{arc.get('current_act')}, tension {arc0.get('tension')}→{arc.get('tension')}; "
          f"now: {(arc.get('next_beats') or [''])[0][:80]}")
    fronts = eng.clocks.fronts(W)
    timed = [h for f in fronts for h in S.history(f.id) if "time passes" in str(h)]
    check("R2 evolving", "the campaign threat advances on its own (time ticks, not player actions)",
          any(int(f.state.get("filled", 0)) > 0 for f in fronts) and bool(timed),
          "; ".join(f"⏱ {f.name[:40]} {f.state.get('filled')}/{f.state.get('segments')}" for f in fronts)
          + f", {len(timed)} time tick(s)")

    # R3 — remembers previously introduced characters and events
    early = [n for n in nodes if n.type == "character" and n.turn <= 2 and n.id != eng.player_id]
    check("R3 memory", "characters introduced at the start are still in memory with their history",
          len(early) >= 2 and all(S.history(n.id) or n.description for n in early),
          ", ".join(f"{n.name} (T{n.turn}, {n.status or 'alive'})" for n in early[:6]))
    segs = {n.turn: n for n in S.nodes(W, types=("segment",), story_id=sid)}
    byid = {n.id: n for n in nodes}
    calls = [(t, [byid[e.dst].name for e in S.edges_of(s.id) if e.rel == "mentions" and e.src == s.id
                  and e.dst in byid and byid[e.dst].turn <= t - 5 and byid[e.dst].type != "player"])
             for t, s in sorted(segs.items()) if t >= 6]
    n_call = sum(bool(c) for _, c in calls)
    check("R3 memory", "later scenes bring back characters/places introduced 5+ turns earlier",
          calls and n_call >= 0.5 * len(calls),
          f"{n_call}/{len(calls)} scenes after turn 5; e.g. " + "; ".join(f"T{t}: {', '.join(c[:3])}" for t, c in calls if c)[:200])

    spec = importlib.util.spec_from_file_location("recall_eval", Path(__file__).parent / "recall_eval.py")
    rec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rec)
    rec.CONFIGS = {"full system": (("named", "dense", "bm25", "relation", "graph"), "gated_ppr"),
                   "plain vector RAG": (("dense",), "ppr")}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):            # evaluate() prints machine-readable RESULT lines
        rec.evaluate(eng, 3, 150, 7)
    rows = {}
    for line in buf.getvalue().splitlines():
        if line.startswith("RESULT\t"):
            _, cname, n, h5, h10, rr, packed = line.split("\t")
            n = int(n)
            rows[cname] = (n, int(h5) / n, int(h10) / n, float(rr) / n, int(packed) / n)
    if "full system" in rows:
        n, h5, h10, mrr, packed = rows["full system"]
        base = rows.get("plain vector RAG", (0, 0, 0, 0, 0))
        check("R3 memory", "probe questions about memories 3+ turns old retrieve the right character/event",
              h5 >= 0.7 and packed >= 0.9,
              f"{n} probes: Hit@5 {h5:.0%}, Hit@10 {h10:.0%}, MRR {mrr:.2f}, reached the prompt {packed:.0%} "
              f"(plain vector RAG: Hit@5 {base[1]:.0%}, MRR {base[3]:.2f})")
    else:
        check("R3 memory", "probe questions retrieve the right memory", False, "recall eval produced no table")

    # bring back an early NPC who has been off-screen, and check the portrayal matches the recorded facts
    recent = {e.dst for t in S.turns(sid, last=3) if t.get("segment_id") for e in S.edges_of(t["segment_id"])
              if e.rel == "mentions"}
    pool = [n for n in early if n.id not in recent and n.status.lower() not in ("dead", "destroyed")] or \
           [n for n in early if n.status.lower() not in ("dead", "destroyed")]
    if not pool:
        check("R3 memory", "an early NPC returns consistent with memory", False, "no living early NPC")
        return
    npc = max(pool, key=lambda n: n.importance)
    facts = [f"{npc.name}: {npc.description}"] + [
        f"{byid[e.src].name if e.src in byid else '?'} {e.rel} {byid[e.dst].name if e.dst in byid else '?'}"
        for e in S.edges_of(npc.id) if e.current and e.rel not in STRUCTURAL_RELS][:6]
    with console.status(f"R3: bringing back {npc.name}…"):
        prep, out = play(eng, f"I go looking for {npc.name} and ask what they have been up to.")
    check("R3 memory", f"{npc.name} (introduced turn {npc.turn}, last seen turn {npc.last_seen_turn}) was recalled into the prompt",
          npc.id in (prep.recall.included if prep.recall else []), f"returned at turn {out.turn}")
    other = next((n for n in nodes if n.type == "character" and n.id not in (npc.id, eng.player_id) and n.description), None)
    decoy = f"{npc.name}: {other.description}" if other else f"{npc.name}: a stranger the hero has never met"
    ok, quote = pick_both(eng.llm, out.prose, f"Which profile matches how {npc.name} is portrayed in this scene?",
                          "; ".join(facts)[:400], decoy[:400])
    check("R3 memory", f"{npc.name} is portrayed as recorded (blind judge vs another character's profile)", ok,
          f"\"{quote[:100]}\"" if ok else "judge did not pick the recorded profile; facts: " + "; ".join(facts)[:150])


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=12, help="total turns in the main story")
    ap.add_argument("--branch-at", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    llm = OllamaLLM()
    ok, msg = llm.health()
    if not ok:
        sys.exit(msg)
    t_start, rng, log = time.time(), random.Random(a.seed), []

    eng = theme_test(llm, data, a.seed)                       # R1; theme 1 becomes the main story
    arc0 = eng.arc.model_dump()
    with console.status(f"Autoplay to turn {a.branch_at}…"):
        autoplay(eng, rng, a.branch_at, log)
    decision_test(eng, data)                                  # R4; branch A continues as the main story
    remaining = a.turns - eng.turn
    for i in range(max(0, remaining)):
        with console.status(f"Autoplay turn {eng.turn + 1}/{a.turns}…"):
            autoplay(eng, rng, 1, log)
    evolving_and_memory(eng, log, arc0)                        # R2 + R3

    table = Table(title="M1-P8 acceptance")
    for col in ("requirement", "check", "result", "evidence"):
        table.add_column(col, overflow="fold")
    for req, name, ok, ev in results:
        table.add_row(req, name, "[green]PASS[/]" if ok else "[red]FAIL[/]", ev)
    console.print(table)
    passed = sum(ok for *_, ok, _ in results)
    console.print(f"[bold]{passed}/{len(results)} checks passed[/] in {(time.time() - t_start) / 60:.1f} min · "
                  f"{llm.stats.calls} LLM calls, constrained-JSON fallbacks {llm.stats.json_failures}/"
                  f"{llm.stats.json_calls} · main story id {eng.story_id} in data/acceptance_theme1.db")
    for req, name, ok, ev in results:
        print(f"RESULT\t{req}\t{'PASS' if ok else 'FAIL'}\t{name}\t{ev}")


if __name__ == "__main__":
    main()
