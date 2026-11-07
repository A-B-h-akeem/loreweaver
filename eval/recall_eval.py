"""Memory-recall benchmark + ablation for Loreweaver.

Generates probe questions automatically from a played story's knowledge graph and measures whether
the retriever brings the right memory back, under several retrieval configurations.

  python eval/recall_eval.py --story 1                       # evaluate an existing story
  python eval/recall_eval.py --play 15 --theme "…"           # autoplay 15 turns first, then evaluate
  python eval/recall_eval.py --story 1 --db path/to/other.db

Probe types (none of them are answerable from the last few scenes alone; targets are >= --min-age turns old):
  name          "What do I remember about Silas?"                 -> the entity
  description   entity description with its name masked             -> the entity   (semantic recall)
  event         "What happened between Silas and Oren at the docks?" -> the event    (multi-hop via graph)
  relationship  "Who hates me?"                                     -> the NPC      (graph edge recall)
  consequence   "What could come back to haunt me with Silas?"      -> the decision (choice -> consequence)
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from loreweaver.config import settings  # noqa: E402
from loreweaver.engine import GameEngine  # noqa: E402
from loreweaver.llm import OllamaLLM  # noqa: E402
from loreweaver.memory import ENTITY_NODE_TYPES, MemoryStore  # noqa: E402
from loreweaver.memory.resolver import content_tokens  # noqa: E402

console = Console()
_BASE = ("named", "dense", "bm25", "relation", "graph")
CONFIGS = {   # name -> (channels, graph scoring mode)
    "phase-2 (named+dense+bm25+relation, PPR)": (_BASE, "ppr"),
    "+ facts (query-to-triple), PPR": (_BASE + ("facts",), "ppr"),
    "gated PPR (query-aware)": (_BASE, "gated_ppr"),
    "spreading activation": (_BASE, "spread"),
    "+ facts + gated PPR": (_BASE + ("facts",), "gated_ppr"),
    "+ fact seeds + gated PPR": (_BASE + ("fact_seeds",), "gated_ppr"),
    "+ facts + spreading activation": (_BASE + ("facts",), "spread"),
    "no graph": (("named", "dense", "bm25", "relation"), "ppr"),
    "dense + bm25 (classic hybrid)": (("dense", "bm25"), "ppr"),
    "dense only (vector RAG)": (("dense",), "ppr"),
}


def mask_names(text: str, names: list[str]) -> str:
    for n in sorted(names, key=len, reverse=True):
        for tok in [n, *content_tokens(n)]:
            if len(tok) >= 3:
                text = re.sub(rf"\b{re.escape(tok)}\b", "someone", text, flags=re.I)
    return text


def build_probes(store: MemoryStore, world_id: int, story_id: int, now: int, player_id: int,
                 min_age: int, rng: random.Random) -> list[dict]:
    probes: list[dict] = []
    nodes = {n.id: n for n in store.nodes(world_id)}
    old = lambda n: n.turn <= now - min_age or n.story_id != story_id  # noqa: E731
    ents = [n for n in nodes.values() if n.type in ENTITY_NODE_TYPES and n.type != "player" and old(n)]
    edges = store.edges(world_id)

    for n in ents:
        probes.append({"type": "name", "q": f"What do I remember about {n.name}?", "target": n.id})
        desc = mask_names(n.description, [n.name, *n.aliases])
        if len(desc.split()) >= 6:
            probes.append({"type": "description", "q": f"The {n.type} described as: {desc}", "target": n.id})

    involves = defaultdict(list)
    where = {}
    for e in edges:
        if e.rel == "involves" and e.dst in nodes and nodes[e.dst].type != "player":
            involves[e.src].append(nodes[e.dst].name)
        if e.rel == "occurred_at" and e.dst in nodes:
            where[e.src] = nodes[e.dst].name
    events = [n for n in nodes.values() if n.type == "event" and n.story_id == story_id and old(n)]
    for n in events:
        if n.importance >= 5 and involves.get(n.id):
            who = involves[n.id][:2]
            at = f" at {where[n.id]}" if n.id in where else ""
            # any event involving the same people (and place) is a correct answer
            ok = {m.id for m in events if set(who) <= set(involves.get(m.id, []))
                  and (not at or where.get(m.id) == where.get(n.id))}
            probes.append({"type": "event", "q": f"What happened with {' and '.join(who)}{at}?", "target": ok})

    to_player = [e for e in edges if e.dst == player_id and e.src in nodes and abs(e.sentiment) >= 3]
    for e in to_player:
        if old(nodes[e.src]):
            ok = {x.src for x in to_player if x.rel == e.rel}        # everyone who "hates" me counts
            probes.append({"type": "relationship", "q": f"Who {e.rel.replace('_', ' ')} me?", "target": ok})

    affects = defaultdict(list)
    for e in edges:
        if e.rel == "affects" and e.dst in nodes:
            affects[e.src].append(nodes[e.dst].name)
    for n in nodes.values():
        if n.type == "decision" and old(n) and affects.get(n.id):
            probes.append({"type": "consequence", "q": f"What could come back to haunt me with "
                                                         f"{' and '.join(affects[n.id][:2])}?", "target": n.id})
    unique = list({p["q"]: p for p in probes}.values())
    rng.shuffle(unique)
    return unique


def evaluate(eng: GameEngine, min_age: int, max_probes: int, seed: int):
    S, W, sid = eng.store, eng.world_id, eng.story_id
    rng = random.Random(seed)
    probes = build_probes(S, W, sid, eng.turn, eng.player_id, min_age, rng)[:max_probes]
    if not probes:
        console.print("[red]No probes (play more turns or lower --min-age).[/]")
        return
    counts = defaultdict(int)
    for p in probes:
        counts[p["type"]] += 1
    console.print(f"Story {sid} at turn {eng.turn}: {len(probes)} probes ("
                  + ", ".join(f"{t} {c}" for t, c in counts.items()) + ")")

    cache: dict[str, object] = {}
    base_embed = eng.retriever.embed_fn

    def cached_embed(texts):
        key = "\x00".join(texts)
        if key not in cache:
            cache[key] = base_embed(texts)
        return cache[key]
    eng.retriever.embed_fn = cached_embed
    titles = {s["id"]: s["title"] for s in S.list_stories(W)}

    results = {}
    for cname, (channels, gmode) in CONFIGS.items():
        per = defaultdict(lambda: {"n": 0, "h5": 0, "h10": 0, "rr": 0.0, "packed": 0})
        for p in probes:
            res = eng.retriever.recall(W, sid, eng.turn + 1, p["q"], focus_ids=[eng.player_id], player_id=eng.player_id,
                                       exclude_ids=[eng.story_node_id], story_titles=titles, channels=channels,
                                       graph_mode=gmode)
            ranked = [r.node.id for r in res.items]
            targets = p["target"] if isinstance(p["target"], set) else {p["target"]}
            rank = next((i + 1 for i, nid in enumerate(ranked) if nid in targets), None)
            for bucket in (p["type"], "ALL"):
                d = per[bucket]
                d["n"] += 1
                d["h5"] += bool(rank and rank <= 5)
                d["h10"] += bool(rank and rank <= 10)
                d["rr"] += 1 / rank if rank else 0
                d["packed"] += bool(targets & set(res.included))
        results[cname] = per
    eng.retriever.embed_fn = base_embed

    types = [t for t in ("name", "description", "event", "relationship", "consequence") if t in results[next(iter(results))]]
    table = Table(title="Memory recall: Hit@5 per probe type, overall Hit@5 / Hit@10 / MRR / packed into prompt")
    table.add_column("configuration")
    for t in types:
        table.add_column(f"{t}\nHit@5", justify="right")
    for c in ("ALL Hit@5", "ALL Hit@10", "MRR", "packed"):
        table.add_column(c, justify="right")
    for cname, per in results.items():
        a = per["ALL"]
        row = [cname] + [f"{per[t]['h5'] / per[t]['n']:.0%}" for t in types]
        row += [f"{a['h5'] / a['n']:.0%}", f"{a['h10'] / a['n']:.0%}", f"{a['rr'] / a['n']:.2f}",
                f"{a['packed'] / a['n']:.0%}"]
        table.add_row(*row, style="bold" if cname.startswith("full") else None)
    console.print(table)
    for cname, per in results.items():   # machine-readable, for aggregating several stories
        a = per["ALL"]
        print(f"RESULT\t{cname}\t{a['n']}\t{a['h5']}\t{a['h10']}\t{a['rr']:.4f}\t{a['packed']}")

    # long-range memory actually used in narration: scenes mentioning entities introduced >= 5 turns earlier
    segs = S.nodes(W, types=("segment",), story_id=sid)
    nodes = {n.id: n for n in S.nodes(W)}
    calls, scenes = 0, 0
    for seg in segs:
        if seg.turn < 6:
            continue
        scenes += 1
        old = [e.dst for e in S.edges_of(seg.id) if e.rel == "mentions" and e.src == seg.id
               and e.dst in nodes and nodes[e.dst].turn <= seg.turn - 5]
        calls += bool(old)
    if scenes:
        console.print(f"Long-range callbacks: {calls}/{scenes} scenes after turn 5 bring back an entity "
                      f"introduced at least 5 turns earlier ({calls / scenes:.0%}).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", type=int)
    ap.add_argument("--play", type=int, default=0)
    ap.add_argument("--theme", default="a smugglers' port where a drowned god is waking")
    ap.add_argument("--db", default=str(settings.db_path))
    ap.add_argument("--min-age", type=int, default=3)
    ap.add_argument("--max-probes", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    eng = GameEngine(MemoryStore(a.db), OllamaLLM(), seed=a.seed)
    if a.story:
        eng.load_story(a.story)
    elif a.play:
        rng = random.Random(a.seed)
        with console.status("Creating story…"):
            eng.new_story(a.theme, "Vex", "Rogue")
            eng.start()
        for i in range(a.play):
            with console.status(f"Autoplay turn {i + 1}/{a.play}…"):
                c = rng.choice(eng.last_outcome.choices)
                eng.play(c.text, c)
        with console.status("Final memory commit…"):
            eng.wait_ready()
    else:
        ap.error("pass --story ID or --play N")
    evaluate(eng, a.min_age, a.max_probes, a.seed)


if __name__ == "__main__":
    main()
