"""In-memory networkx view of a world's story graph, with Personalized PageRank.

PPR spreads relevance from the seed nodes (what the player is doing or talking about) along
relations, co-occurrence and causality. That surfaces connected memories the query never mentioned.
"""
from __future__ import annotations

import networkx as nx

from .store import MemoryStore

# How strongly each edge kind conducts relevance during PageRank.
REL_CONDUCTANCE = {"mentions": 0.5, "next": 0.15, "summarizes": 0.3, "appears_in": 0.1,
                   "from_scene": 0.3, "made_in": 0.3}


class StoryGraph:
    def __init__(self, store: MemoryStore):
        self.store = store
        self._cache: dict[int, tuple[int, nx.Graph]] = {}

    def graph(self, world_id: int) -> nx.Graph:
        cached = self._cache.get(world_id)
        if cached and cached[0] == self.store.version:
            return cached[1]
        g = nx.Graph()
        for n in self.store.nodes(world_id):
            g.add_node(n.id, type=n.type, name=n.name, importance=n.importance, story_id=n.story_id)
        for e in self.store.edges(world_id):
            if e.src not in g or e.dst not in g:
                continue
            w = e.weight * REL_CONDUCTANCE.get(e.rel, 1.0) * (1.0 if e.current else 0.3)   # history conducts less
            if g.has_edge(e.src, e.dst):          # collapse parallel relations, keep labels
                g[e.src][e.dst]["weight"] += w
                g[e.src][e.dst]["rels"].append(e.rel)
            else:
                g.add_edge(e.src, e.dst, weight=w, rels=[e.rel])
        self._cache[world_id] = (self.store.version, g)
        return g

    def ppr(self, world_id: int, seeds: dict[int, float], alpha: float = 0.85) -> dict[int, float]:
        g = self.graph(world_id)
        pers = {n: w for n, w in seeds.items() if n in g and w > 0}
        if not pers or g.number_of_edges() == 0:
            return {n: w for n, w in pers.items()}
        try:
            return nx.pagerank(g, alpha=alpha, personalization=pers, weight="weight", max_iter=200, tol=1e-8)
        except nx.PowerIterationFailedConvergence:
            return pers

    def gated_ppr(self, world_id: int, seeds: dict[int, float], gate: dict[int, float],
                  floor: float = 0.2, alpha: float = 0.85) -> dict[int, float]:
        """Query-aware PageRank: an edge conducts relevance in proportion to how query-relevant its endpoints
        are, w' = w * (floor + mean gate). Gate σ(v) = max(cos(e_v, e_q), 0) (after arXiv 2606.30133)."""
        g = self.graph(world_id)
        pers = {n: w for n, w in seeds.items() if n in g and w > 0}
        if not pers or g.number_of_edges() == 0:
            return dict(pers)
        h = nx.Graph()
        h.add_nodes_from(g.nodes)
        for u, v, d in g.edges(data=True):
            gu, gv = max(gate.get(u, 0.0), 0.0), max(gate.get(v, 0.0), 0.0)
            h.add_edge(u, v, weight=d["weight"] * (floor + (gu + gv) / 2))
        try:
            return nx.pagerank(h, alpha=alpha, personalization=pers, weight="weight", max_iter=200, tol=1e-8)
        except nx.PowerIterationFailedConvergence:
            return pers

    def spread(self, world_id: int, seeds: dict[int, float], gate: dict[int, float],
               steps: int = 3, decay: float = 0.7, floor: float = 0.1) -> dict[int, float]:
        """Query-aware spreading activation (arXiv 2606.30133): T fixed steps, activation flows only from
        higher- to lower-activation neighbours, scaled by decay and the receiving node's query gate."""
        g = self.graph(world_id)
        act = {n: w for n, w in seeds.items() if n in g and w > 0}
        if not act:
            return {}
        top = max(act.values())
        act = {n: w / top for n, w in act.items()}
        for _ in range(steps):
            delta: dict[int, float] = {}
            for u, ru in act.items():
                nbrs = g[u]
                total_w = sum(d["weight"] for d in nbrs.values()) or 1.0
                for v, d in nbrs.items():
                    if act.get(v, 0.0) >= ru:                    # monotone: only downhill
                        continue
                    sv = floor + max(gate.get(v, 0.0), 0.0)
                    delta[v] = delta.get(v, 0.0) + decay * sv * ru * d["weight"] / total_w
            if not delta:
                break
            for v, dv in delta.items():
                act[v] = act.get(v, 0.0) + dv
        return act

    def neighbors(self, world_id: int, node_id: int, depth: int = 1) -> set[int]:
        g = self.graph(world_id)
        if node_id not in g:
            return set()
        return set(nx.single_source_shortest_path_length(g, node_id, cutoff=depth)) - {node_id}

    def path(self, world_id: int, a: int, b: int) -> list[int]:
        """Shortest connection between two memories, e.g. how a stranger is tied to a past betrayal."""
        try:
            return nx.shortest_path(self.graph(world_id), a, b)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
