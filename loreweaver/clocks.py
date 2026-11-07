"""Consequence clocks (PLAN.md §8C), after Blades in the Dark progress clocks and Dungeon World fronts.

A clock is a node with N segments and a concrete payoff. Scenes, failed rolls and the passage of time
tick it. When it fills, code (not the LLM) declares it due, and the next scene must deliver the payoff.
That makes consequences concrete and stops them "resolving" before they actually happen.
"""
from __future__ import annotations

import difflib
import re
from typing import Iterable, Optional

from .config import settings
from .memory import MemoryStore, Node

SEGMENTS_BY_SEVERITY = {1: 4, 2: 4, 3: 6, 4: 6, 5: 8}
ACTIVE = ("ticking", "due")


def _sim(a: str, b: str) -> float:
    ta, tb = set(re.findall(r"[a-z]{3,}", a.lower())), set(re.findall(r"[a-z]{3,}", b.lower()))
    return max(len(ta & tb) / max(1, len(ta | tb)), difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio())


def bar(node: Node) -> str:
    n, f = int(node.state.get("segments", 4)), int(node.state.get("filled", 0))
    return "■" * min(f, n) + "□" * max(0, n - f)


class Clocks:
    def __init__(self, store: MemoryStore):
        self.store = store

    # ------------------------------------------------------------------ queries
    def active(self, world_id: int) -> list[Node]:
        out = [n for n in self.store.nodes(world_id, types=("clock",)) if n.state.get("status") in ACTIVE]
        out.sort(key=lambda n: -int(n.state.get("filled", 0)) / max(1, int(n.state.get("segments", 4))))
        return out

    def due(self, world_id: int) -> list[Node]:
        return sorted((n for n in self.active(world_id) if n.state.get("status") == "due"),
                      key=lambda n: n.state.get("due_turn", 0))

    def fronts(self, world_id: int) -> list[Node]:
        return [n for n in self.active(world_id) if n.state.get("front")]

    def match(self, clocks: Iterable[Node], name: str, threshold: float = 0.5) -> Optional[Node]:
        best = max(clocks, key=lambda c: _sim(c.name, name), default=None)
        return best if best is not None and _sim(best.name, name) >= threshold else None

    def most_relevant(self, world_id: int, entity_ids: set[int]) -> Optional[Node]:
        """The danger clock threatening someone in the scene, else a front, else the fullest clock."""
        clocks = [c for c in self.active(world_id) if c.state.get("status") == "ticking"]
        if not clocks:
            return None
        for c in clocks:
            threatened = {e.dst for e in self.store.edges_of(c.id) if e.rel == "threatens" and e.src == c.id}
            if threatened & entity_ids:
                return c
        fronts = [c for c in clocks if c.state.get("front")]
        return (fronts or clocks)[0]

    # ------------------------------------------------------------------ mutations
    def create(self, world_id: int, story_id: int, turn: int, name: str, payoff: str, segments: int, *,
               kind: str = "danger", threatens: Iterable[int] = (), decision_id: int | None = None,
               portents: list[str] | None = None, front: bool = False, story_node_id: int | None = None) -> int:
        state = {"kind": kind, "segments": segments, "filled": 0, "payoff": payoff, "status": "ticking",
                 "front": front}
        if portents:
            state["portents"] = portents[:4]
        cid = self.store.add_node(world_id, story_id, "clock", name[:80], description=payoff, state=state,
                                  importance=8 if front else 6, turn=turn)
        for t in threatens:
            self.store.upsert_edge(world_id, story_id, cid, t, "threatens", turn=turn)
        if decision_id:
            self.store.upsert_edge(world_id, story_id, decision_id, cid, "started", turn=turn)
        if story_node_id:
            self.store.upsert_edge(world_id, story_id, cid, story_node_id, "appears_in", turn=turn)
        return cid

    def tick(self, clock_id: int, n: int, story_id: int, turn: int, reason: str = "",
             source: str = "scene") -> tuple[Node, list[str]]:
        """Advance a clock; returns (node, notes). A full clock becomes `due`.

        Pacing (learned from a real run in which a front fired by turn 6): scene ticks respect a cooldown
        (2 turns, or 3 for fronts), and a front cannot fill its last segment before `front_min_turn`, so the
        campaign doom stays a climax instead of an early accident. Time and failure ticks skip the cooldown.
        """
        c = self.store.get_node(clock_id)
        if c is None or c.state.get("status") != "ticking" or n <= 0:
            return c, []
        front = bool(c.state.get("front"))
        last = c.state.get("last_scene_tick")              # the cooldown limits Chronicler ticks only
        if source == "scene" and last is not None and turn - int(last) < (4 if front else 2):
            return c, []
        if source == "scene" and front:
            n = 1                                            # scenes nudge the front; time drives it
        seg, before = int(c.state.get("segments", 4)), int(c.state.get("filled", 0))
        after = min(seg, before + n)
        if front and after >= seg and turn < settings.front_min_turn:
            after = seg - 1                                  # the doom looms, but waits for the finale
        if after == before:
            return c, []
        state = {"filled": after, "last_reason": reason[:160], "last_tick_turn": turn}
        if source == "scene":
            state["last_scene_tick"] = turn
        notes = [f"⏱ {c.name} {bar_from(after, seg)}" + (f" ({reason[:60]})" if reason else "")]
        portents = c.state.get("portents") or []
        if portents:                                    # fronts reveal a grim portent at even fractions
            step = seg / (len(portents) + 1)
            for i, p in enumerate(portents, 1):
                if before < round(step * i) <= after:
                    state["current_portent"] = p
                    notes.append(f"Grim portent: {p}")
        if after >= seg:
            state.update(status="due", due_turn=turn)
            notes.append(f"⚠ {c.name} is DUE: {c.state.get('payoff', '')[:80]}")
        return self.store.update_node(clock_id, story_id, turn, state=state), notes

    def fire(self, clock_id: int, world_id: int, story_id: int, turn: int, event_ids: Iterable[int]) -> list[str]:
        c = self.store.update_node(clock_id, story_id, turn, state={"status": "fired", "fired_turn": turn})
        event_ids = list(event_ids)
        for eid in event_ids:
            self.store.upsert_edge(world_id, story_id, clock_id, eid, "led_to", turn=turn)
        for e in self.store.edges_of(clock_id):
            if e.rel == "started" and e.dst == clock_id:      # the decision that set it in motion
                self.store.update_node(e.src, story_id, turn, state={"status": "resolved", "resolved_turn": turn})
                for eid in event_ids:
                    self.store.upsert_edge(world_id, story_id, e.src, eid, "led_to", turn=turn)
        return [f"Consequence landed: {c.name}"]

    def avert(self, clock_id: int, story_id: int, turn: int, quote: str) -> list[str]:
        c = self.store.update_node(clock_id, story_id, turn, state={"status": "averted", "averted_turn": turn,
                                                                    "averted_by": quote[:200]})
        for e in self.store.edges_of(clock_id):
            if e.rel == "started" and e.dst == clock_id:
                self.store.update_node(e.src, story_id, turn, state={"status": "averted"})
        return [f"Averted: {c.name}"]


def bar_from(filled: int, segments: int) -> str:
    return "■" * filled + "□" * max(0, segments - filled)


def describe(node: Node) -> str:
    s = node.state
    return (f"{node.name} [{bar(node)} {s.get('filled', 0)}/{s.get('segments', 4)}] → {s.get('payoff', '')}"
            + (f" (now: {s['current_portent']})" if s.get("current_portent") else ""))
