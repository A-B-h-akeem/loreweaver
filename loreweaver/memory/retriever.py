"""Hybrid GraphRAG retrieval (PLAN.md §3.3).

exact-name seeds + dense (cosine) + sparse (BM25) -> Reciprocal Rank Fusion -> Personalized PageRank
over the story graph -> scoring (relevance, graph, importance, recency, boosts) -> packing into
sections within a token budget.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

from ..config import settings
from .graph import StoryGraph
from .resolver import GENERIC, content_tokens, normalize
from .store import ENTITY_NODE_TYPES, MemoryStore, Node

QUERY_INSTRUCT = "Instruct: Given the current scene of an interactive story, retrieve memories " \
                 "(characters, places, past events, decisions) relevant to it\nQuery: "

W_REL, W_GRAPH, W_IMP, W_REC = 0.40, 0.30, 0.15, 0.15
ALL_CHANNELS = ("named", "dense", "bm25", "relation", "facts", "graph")
NEG_WORDS = {"hate", "hates", "hated", "hostile", "enemy", "enemies", "angry", "grudge", "distrust", "distrusts",
             "fear", "fears", "despise", "despises", "resent", "resents", "hunting", "hunts", "betray", "betrayed"}
POS_WORDS = {"like", "likes", "love", "loves", "trust", "trusts", "ally", "allies", "friend", "friends",
             "loyal", "grateful", "admire", "admires", "owe", "owes", "help", "helped"}
FOCUS_SEED = 0.2          # weight of the hero + current location as PageRank seeds
STRUCTURAL_RELS = {"mentions", "next", "summarizes", "appears_in", "involves", "occurred_at", "concerns",
                   "affects"}


def est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class Recalled:
    node: Node
    score: float
    why: list[str] = field(default_factory=list)
    parts: dict = field(default_factory=dict)


@dataclass
class RecallResult:
    query: str
    items: list[Recalled]
    context: str
    included: list[int]

    def to_log(self) -> dict:
        inc = set(self.included)
        return {"query": self.query[:500],
                "items": [{"id": r.node.id, "name": r.node.name[:80], "type": r.node.type,
                           "score": round(r.score, 3), "why": r.why, "included": r.node.id in inc,
                           **{k: round(v, 3) for k, v in r.parts.items()}} for r in self.items[:40]]}


class MemoryRetriever:
    def __init__(self, store: MemoryStore, graph: StoryGraph,
                 embed_fn: Callable[[list[str]], np.ndarray] | None = None):
        self.store = store
        self.graph = graph
        self.embed_fn = embed_fn
        self._vec_cache: dict[int, tuple[int, list[int], np.ndarray]] = {}
        self._name_cache: dict[int, tuple[int, list[tuple[int, re.Pattern, float]]]] = {}
        self._edge_cache: dict[int, tuple[int, list]] = {}
        self._fact_vecs: dict[tuple, np.ndarray] = {}      # fact-sentence embeddings keyed by edge content
        self._qcache: Optional[tuple[str, np.ndarray]] = None

    # ------------------------------------------------------------------ channels
    def _matrix(self, world_id: int) -> tuple[list[int], np.ndarray]:
        c = self._vec_cache.get(world_id)
        if c and c[0] == self.store.version:
            return c[1], c[2]
        ids, mat = self.store.embedding_matrix(world_id)
        self._vec_cache[world_id] = (self.store.version, ids, mat)
        return ids, mat

    def _query_vec(self, query: str) -> Optional[np.ndarray]:
        if not self.embed_fn:
            return None
        if self._qcache and self._qcache[0] == query:
            return self._qcache[1]
        try:
            q = self.embed_fn([QUERY_INSTRUCT + query])[0]
        except Exception:  # noqa: BLE001 - embedding server hiccup must not break a turn
            return None
        self._qcache = (query, q)
        return q

    def node_sims(self, world_id: int, query: str) -> dict[int, float]:
        """Cosine similarity of the query to every embedded node (one matrix product)."""
        ids, mat = self._matrix(world_id)
        q = self._query_vec(query)
        if q is None or not ids or q.shape[0] != mat.shape[1]:
            return {}
        sims = mat @ q
        return {ids[i]: float(sims[i]) for i in range(len(ids))}

    def dense(self, world_id: int, query: str, k: int) -> list[tuple[int, float]]:
        sims = self.node_sims(world_id, query)
        return sorted(sims.items(), key=lambda x: -x[1])[:k]

    def fact_hits(self, world_id: int, query: str, player_id: int | None, k: int = 6) -> dict[int, float]:
        """HippoRAG-2-style query-to-triple matching: relation edges embedded as fact sentences
        ("Silas Crane hates Kael (the hero, me)"); the best-matching facts seed both endpoints."""
        q = self._query_vec(query)
        if q is None:
            return {}
        edges = [e for e in self._edges(world_id)
                 if e.current and e.rel not in STRUCTURAL_RELS and e.rel != "located_at"]
        if not edges:
            return {}
        names = {n.id: n.name for n in self.store.get_nodes({e.src for e in edges} | {e.dst for e in edges}).values()}

        def label(nid: int) -> str:
            return f"{names.get(nid, '?')} (the hero, me)" if nid == player_id else names.get(nid, "?")

        texts = {(e.src, e.dst, e.rel, e.sentiment): f"{label(e.src)} {e.rel.replace('_', ' ')} {label(e.dst)}"
                 for e in edges}
        missing = [key for key in texts if key not in self._fact_vecs]
        if missing:
            try:
                vecs = self.embed_fn([texts[key] for key in missing])
            except Exception:  # noqa: BLE001
                return {}
            self._fact_vecs.update({key: vecs[i] for i, key in enumerate(missing)})
        keys = list(texts)
        mat = np.vstack([self._fact_vecs[key] for key in keys])
        if mat.shape[1] != q.shape[0]:
            return {}
        sims = mat @ q
        order = np.argsort(-sims)[:k]
        top = float(sims[order[0]]) if len(order) else 0.0
        hits: dict[int, float] = {}
        for i in order:
            if sims[i] < max(0.3, top - 0.15):                 # keep only facts close to the best match
                break
            w = float(sims[i]) / max(top, 1e-6)
            src, dst = keys[i][0], keys[i][1]
            hits[src] = max(hits.get(src, 0.0), w)
            hits[dst] = max(hits.get(dst, 0.0), 0.8 * w)
        return hits

    def _name_patterns(self, world_id: int) -> list[tuple[int, re.Pattern, float]]:
        c = self._name_cache.get(world_id)
        if c and c[0] == self.store.version:
            return c[1]
        pats = []
        for n in self.store.nodes(world_id, types=ENTITY_NODE_TYPES):
            if n.type == "player":
                continue
            for nm in {normalize(x) for x in [n.name, *n.aliases]}:
                if len(nm) >= 3 and nm not in GENERIC:
                    pats.append((n.id, re.compile(rf"\b{re.escape(nm)}\b"), 1.0))
            for tok in content_tokens(n.name):  # "Silas" from "Silas Crane"
                if len(tok) >= 4 and tok not in GENERIC:
                    pats.append((n.id, re.compile(rf"\b{re.escape(tok)}\b"), 0.6))
        self._name_cache[world_id] = (self.store.version, pats)
        return pats

    def named(self, world_id: int, query: str) -> dict[int, float]:
        q = normalize(query)
        hits: dict[int, float] = {}
        for nid, pat, w in self._name_patterns(world_id):
            if pat.search(q):
                hits[nid] = max(hits.get(nid, 0.0), w)
        return hits

    def relation_hits(self, world_id: int, query: str, named: dict[int, float],
                      player_id: int | None) -> dict[int, float]:
        """Relation-aware channel: relationships live on edges, which dense/BM25 over node text can't see.

        "Who distrusts me?" -> sources of `distrusts` edges pointing at the hero;
        "Who hates me?"     -> also anyone with strongly negative sentiment toward the hero.
        """
        q = query.lower()
        stems = {w[:5] for w in re.findall(r"[a-z]{4,}", q)}
        about_me = player_id is not None and re.search(r"\b(me|my|i|myself|mine)\b", q) is not None
        want_neg = bool(NEG_WORDS & set(re.findall(r"[a-z]+", q)))
        want_pos = bool(POS_WORDS & set(re.findall(r"[a-z]+", q)))
        if not (stems or want_neg or want_pos):
            return {}
        anchors = set(named) | ({player_id} if about_me else set())
        if not anchors:
            return {}
        hits: dict[int, float] = {}
        for e in self._edges(world_id):
            if not e.current or e.rel in STRUCTURAL_RELS or (e.src not in anchors and e.dst not in anchors):
                continue
            label_match = bool({w[:5] for w in re.findall(r"[a-z]{4,}", e.rel.replace("_", " "))} & stems)
            mood_match = (want_neg and e.sentiment <= -3) or (want_pos and e.sentiment >= 3)
            if not (label_match or mood_match):
                continue
            w = 1.0 if label_match else 0.7
            if e.dst in anchors and e.src not in anchors:      # "who X me"  -> the subject
                hits[e.src] = max(hits.get(e.src, 0.0), w)
            elif e.src in anchors and e.dst not in anchors:    # "whom do I X" -> the object
                hits[e.dst] = max(hits.get(e.dst, 0.0), 0.8 * w)
        return hits

    def _edges(self, world_id: int):
        c = self._edge_cache.get(world_id)
        if c and c[0] == self.store.version:
            return c[1]
        edges = self.store.edges(world_id)
        self._edge_cache[world_id] = (self.store.version, edges)
        return edges

    # ------------------------------------------------------------------ main entry
    def recall(self, world_id: int, story_id: int, turn: int, query: str, *,
               focus_ids: Iterable[int] = (), exclude_ids: Iterable[int] = (),
               budget_tokens: int | None = None, story_titles: dict[int, str] | None = None,
               player_id: int | None = None, channels: Iterable[str] | None = None,
               graph_mode: str | None = None) -> RecallResult:
        """`channels` and `graph_mode` exist for ablation studies (eval/recall_eval.py); the game uses the
        configured defaults (settings.retrieval_channels, settings.graph_mode)."""
        budget = budget_tokens or settings.memory_token_budget
        k = settings.retrieve_candidates
        exclude = set(exclude_ids)
        channels = set(channels if channels is not None else settings.retrieval_channels)
        graph_mode = graph_mode or settings.graph_mode

        named = self.named(world_id, query) if "named" in channels else {}
        dense = self.dense(world_id, query, k) if "dense" in channels else []
        sparse = self.store.fts_search(world_id, query, k) if "bm25" in channels else []
        relational = self.relation_hits(world_id, query, named, player_id) if "relation" in channels else {}
        facts = self.fact_hits(world_id, query, player_id) if channels & {"facts", "fact_seeds"} else {}
        if "facts" in channels:                    # fact matches behave like relation matches (boost + seed)
            for nid, w in facts.items():
                relational[nid] = max(relational.get(nid, 0.0), 0.8 * w)
        fact_seeds = facts if "fact_seeds" in channels else {}   # HippoRAG-2 style: seeds for the walk only

        rrf: dict[int, float] = {}
        ranks: dict[int, list[str]] = {}
        for label, results in (("dense", dense), ("bm25", sparse)):
            for rank, (nid, _) in enumerate(results):
                rrf[nid] = rrf.get(nid, 0.0) + 1.0 / (settings.rrf_k + rank + 1)
                ranks.setdefault(nid, []).append(f"{label}#{rank + 1}")
        rrf_max = max(rrf.values(), default=1.0)
        rel = {nid: v / rrf_max for nid, v in rrf.items()}

        seeds: dict[int, float] = {nid: v for nid, v in rel.items()}
        for nid, w in named.items():
            seeds[nid] = seeds.get(nid, 0.0) + 1.5 * w
        for nid, w in relational.items():
            seeds[nid] = seeds.get(nid, 0.0) + 1.0 * w
        for nid, w in fact_seeds.items():
            seeds[nid] = seeds.get(nid, 0.0) + 0.5 * w
        for nid in focus_ids:
            seeds[nid] = seeds.get(nid, 0.0) + FOCUS_SEED
        ppr: dict[int, float] = {}
        if "graph" in channels:
            if graph_mode == "ppr":
                ppr = self.graph.ppr(world_id, seeds)
            else:
                gate = self.node_sims(world_id, query)
                ppr = (self.graph.gated_ppr(world_id, seeds, gate) if graph_mode == "gated_ppr"
                       else self.graph.spread(world_id, seeds, gate))
        # PageRank favours hubs (the hero, the main location). Dividing by sqrt(degree) keeps the score
        # about *specific* connection to the seeds rather than general centrality.
        g = self.graph.graph(world_id) if ppr else None
        ppr = {n: v / math.sqrt(max(1, g.degree(n))) for n, v in ppr.items()} if g is not None else {}
        ppr_max = max((v for n, v in ppr.items() if n not in focus_ids), default=0.0) or 1.0
        graph_s = {n: min(1.0, v / ppr_max) for n, v in ppr.items()}

        cand = set(rel) | set(named) | set(relational) | \
            {n for n, _ in sorted(graph_s.items(), key=lambda x: -x[1])[:50]}
        cand -= exclude
        if player_id is not None:
            cand.discard(player_id)
        nodes = self.store.get_nodes(cand)

        items: list[Recalled] = []
        for nid, node in nodes.items():
            if node.type == "story":                              # hub nodes, not memories
                continue
            if node.last_seen_story == story_id:
                rec = math.exp(-max(0, turn - node.last_seen_turn) / 12.0)
            else:
                rec = 0.15                                      # memory from another story in this world
            parts = {"rel": rel.get(nid, 0.0), "graph": graph_s.get(nid, 0.0),
                     "imp": node.importance / 10.0, "rec": rec}
            score = W_REL * parts["rel"] + W_GRAPH * parts["graph"] + W_IMP * parts["imp"] + W_REC * parts["rec"]
            why = list(ranks.get(nid, []))
            if nid in named:
                score += 0.35 * named[nid]
                why.insert(0, "named")
            if nid in relational:
                score += 0.35 * relational[nid]
                why.insert(0, "fact" if nid in facts and relational[nid] <= 0.8 * facts[nid] + 1e-9 else "relation")
            if parts["graph"] > 0.15 and nid not in named:
                why.append("graph")
            if node.type == "thread" and node.status in ("opened", "advanced", "open"):
                score += 0.10
                why.append("open-thread")
            if node.type == "decision" and node.status == "pending":
                score += 0.12
                why.append("pending-consequence")
            if node.type == "clock" and node.state.get("status") in ("ticking", "due"):
                fill = int(node.state.get("filled", 0)) / max(1, int(node.state.get("segments", 4)))
                score += 0.08 + 0.15 * fill                      # the closer to firing, the more it matters
                why.append("ticking-clock")
            elif node.type == "clock":
                score -= 0.2                                     # fired/averted clocks are history
            if node.type == "segment":
                score -= 0.05
            if node.story_id not in (None, story_id):
                why.append("other-story")
            items.append(Recalled(node, score, why, parts))
        items.sort(key=lambda r: -r.score)
        context, included = self._pack(items, budget, story_id, story_titles or {}, query, player_id)
        return RecallResult(query=query, items=items, context=context, included=included)

    # ------------------------------------------------------------------ packing
    def _entity_card(self, node: Node, story_id: int, titles: dict[int, str], player_id: Optional[int]) -> str:
        bits = [node.type]
        st = node.state
        if st.get("status"):
            bits.append(f"status: {st['status']}")
        if st.get("attitude") and node.type != "player":
            bits.append(f"attitude: {st['attitude']}")
        if st.get("location"):
            bits.append(f"at {st['location']}")
        card = f"• {node.name} ({'; '.join(bits)})"
        if node.aliases:
            card += f" aka {', '.join(node.aliases[:3])}"
        if node.description:
            card += f" — {node.description[:220]}"
        # one phrase per (counterpart, direction): "distrusts/hates the hero (-8)"
        groups: dict[tuple[int, bool], list] = {}
        for e in self.store.edges_of(node.id):
            if e.rel in STRUCTURAL_RELS or not e.current:          # superseded facts are history, not state
                continue
            outgoing = e.src == node.id
            groups.setdefault((e.dst if outgoing else e.src, outgoing), []).append(e)
        ranked = sorted(groups.items(), key=lambda kv: -max(abs(e.sentiment) + e.weight for e in kv[1]))[:5]
        others = self.store.get_nodes([oid for (oid, _), _ in ranked])
        rels = []
        for (oid, outgoing), es in ranked:
            o = others.get(oid)
            if not o:
                continue
            es.sort(key=lambda e: -e.updated_turn)
            labels = "/".join(dict.fromkeys(e.rel.replace("_", " ") for e in es[:3]))
            oname = "the hero" if oid == player_id else o.name
            phrase = f"{labels} {oname}" if outgoing else f"{oname} {labels} them"
            sentiment = es[0].sentiment or next((e.sentiment for e in es if e.sentiment), 0)
            if sentiment:
                phrase += f" ({sentiment:+d})"
            rels.append(phrase)
        if rels:
            card += " | " + "; ".join(rels)
        card += f" [introduced turn {node.turn}"
        if node.story_id not in (None, story_id):
            card += f" of the past tale '{titles.get(node.story_id, 'another story')}'"
        card += "]"
        return card

    @staticmethod
    def _excerpt(text: str, query: str, max_chars: int = 420) -> str:
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        terms = set(re.findall(r"[a-z]{4,}", query.lower()))
        scored = sorted(range(len(sents)), key=lambda i: -len(terms & set(re.findall(r"[a-z]{4,}", sents[i].lower()))))
        keep = sorted(scored[:3])
        out = " … ".join(sents[i] for i in keep)
        return out[:max_chars]

    def _pack(self, items: list[Recalled], budget: int, story_id: int, titles: dict[int, str],
              query: str, player_id: Optional[int]) -> tuple[str, list[int]]:
        caps = {"entities": int(budget * 0.45), "events": int(budget * 0.30),
                "threads": int(budget * 0.10), "passages": int(budget * 0.15)}
        used = {k: 0 for k in caps}
        out: dict[str, list[tuple[int, str]]] = {k: [] for k in caps}
        included: list[int] = []
        for r in items:
            n = r.node
            if r.score < 0.12:
                break
            if n.is_entity:
                sec, line, order = "entities", self._entity_card(n, story_id, titles, player_id), 0
            elif n.type == "event":
                tag = f"T{n.turn}" if n.story_id == story_id else f"past tale '{titles.get(n.story_id, '?')}'"
                sec, line, order = "events", f"[{tag}] {n.description or n.name}", n.turn
            elif n.type in ("thread", "decision", "clock"):
                if n.type == "thread":
                    line = f"• Plot thread: {n.name} ({n.status or 'open'}) — {n.description[:160]}"
                elif n.type == "clock":
                    st = n.state
                    line = (f"• Looming threat: {n.name} [{st.get('filled', 0)}/{st.get('segments', 4)}, "
                            f"{st.get('status', 'ticking')}] → {st.get('payoff', '')[:160]}")
                else:
                    line = (f"• Hero's past decision (T{n.turn}): {n.name} → possible consequence: "
                            f"{n.state.get('consequence', '')[:160]} [{n.status or 'pending'}]")
                sec, order = "threads", n.turn
            else:
                label = "Chapter summary" if n.type == "summary" else f"Scene T{n.turn}"
                sec, line, order = "passages", f"({label}) {self._excerpt(n.content or n.description, query)}", n.turn
            cost = est_tokens(line)
            if used[sec] + cost > caps[sec]:
                continue
            used[sec] += cost
            out[sec].append((order, line))
            included.append(n.id)

        titles_map = {"entities": "Characters, places & things involved",
                      "events": "Relevant past events (chronological)",
                      "threads": "Plot threads & consequences of the hero's choices",
                      "passages": "Recalled passages"}
        blocks = []
        for sec in ("entities", "events", "threads", "passages"):
            lines = out[sec]
            if not lines:
                continue
            if sec != "entities":
                lines = sorted(lines, key=lambda x: x[0])
            blocks.append(f"### {titles_map[sec]}\n" + "\n".join(l for _, l in lines))
        return "\n\n".join(blocks) or "(no relevant memories yet)", included
