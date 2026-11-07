"""Entity resolution: map a surface mention ("Harbormaster Silas", "the old hermit") to an existing node.

Cascade: exact name/alias -> distinctive token-subset -> fuzzy string -> embedding similarity.
Without it the graph fills with duplicates and recall degrades quickly.
"""
from __future__ import annotations

import difflib
import re
from typing import Callable, Optional

import numpy as np

from ..config import settings
from .store import ENTITY_NODE_TYPES, MemoryStore, Node

TITLES = set("""the a an of lord lady sir dame captain capt master mistress mister mr mrs ms miss dr doctor
king queen prince princess duke duchess baron baroness count countess father mother brother sister
old young elder high saint st harbormaster general commander sergeant lieutenant professor
""".split())
GENERIC = set("""guard guards cultist cultists man woman men women stranger merchant soldier figure priest
child boy girl person people villager villagers thug thugs bandit bandits creature beast monster
shadow voice crowd officer agent leader member servant innkeeper bartender barkeep sailor pirate
knight mage wizard witch hunter thief traveler traveller figure door room street road
fighter rogue cleric ranger bard warrior hero heroine adventurer""".split())
PLAYER_WORDS = {"you", "player", "the player", "hero", "the hero", "adventurer", "protagonist", "yourself",
                "pc", "the adventurer", "player character"}


def normalize(name: str) -> str:
    n = re.sub(r"[^\w\s'-]", " ", name.lower()).replace("'s ", " ")
    n = re.sub(r"^(the|a|an)\s+", "", n.strip())
    return re.sub(r"\s+", " ", n).strip()


_NUM = re.compile(r"\b(\d+|ii|iii|iv|vi|vii|viii|ix)\b")


def numbers_conflict(a: str, b: str) -> bool:
    """'Cult Guard 1' vs 'Cult Guard 2' (or 'Edric II' vs 'Edric III') are different individuals."""
    na, nb = set(_NUM.findall(a.lower())), set(_NUM.findall(b.lower()))
    return bool(na or nb) and na != nb


def content_tokens(name: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9']+", normalize(name)) if t not in TITLES and len(t) > 1}


class EntityResolver:
    def __init__(self, store: MemoryStore, embed_fn: Callable[[list[str]], np.ndarray] | None = None):
        self.store = store
        self.embed_fn = embed_fn
        self._cache: dict[int, tuple[int, list[tuple[Node, set[str]]]]] = {}

    def _entities(self, world_id: int) -> list[tuple[Node, set[str]]]:
        c = self._cache.get(world_id)
        if c and c[0] == self.store.version:
            return c[1]
        items = [(n, {normalize(x) for x in [n.name, *n.aliases]})
                 for n in self.store.nodes(world_id, types=ENTITY_NODE_TYPES)]
        self._cache[world_id] = (self.store.version, items)
        return items

    def is_player_ref(self, name: str, player_name: str) -> bool:
        n = normalize(name)
        return n in PLAYER_WORDS or n == normalize(player_name) or (
            bool(content_tokens(player_name)) and content_tokens(name) == content_tokens(player_name))

    def resolve(self, world_id: int, name: str, type: str | None = None, description: str = "",
                aliases: list[str] | None = None) -> Optional[Node]:
        mentions = [m for m in [name, *(aliases or [])] if m and m.strip()]
        ents = self._entities(world_id)
        if not ents or not mentions:
            return None

        def compatible(n: Node) -> bool:
            return type is None or n.type == type or "lore" in (n.type, type) or n.type == "player"

        # 1) exact name or alias
        for m in mentions:
            nm = normalize(m)
            hits = [n for n, names in ents if nm in names]
            hits = [n for n in hits if compatible(n)] or hits
            if hits:
                return max(hits, key=lambda n: n.last_seen_turn)

        # 2) distinctive token subset ("Silas" <-> "Harbormaster Silas Crane")
        for m in mentions:
            mt = content_tokens(m)
            if not mt or mt <= GENERIC:
                continue
            hits = []
            for n, names in ents:
                if not compatible(n) or n.type == "player":
                    continue
                for nm in names:
                    nt = content_tokens(nm)
                    if not nt or nt <= GENERIC or numbers_conflict(m, nm):
                        continue
                    small, big = (mt, nt) if len(mt) <= len(nt) else (nt, mt)
                    if small <= big and (small - GENERIC):
                        hits.append(n)
                        break
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:  # ambiguous: prefer same type, then most recently seen
                same = [h for h in hits if h.type == type] or hits
                return max(same, key=lambda n: (n.last_seen_turn, n.importance))

        # 3) fuzzy string (typos, "Ser Aldric" vs "Sir Aldric")
        best, best_r = None, 0.0
        for m in mentions:
            nm = normalize(m)
            for n, names in ents:
                if not compatible(n) or n.type == "player":
                    continue
                for x in names:
                    if numbers_conflict(nm, x):
                        continue
                    r = difflib.SequenceMatcher(None, nm, x).ratio()
                    if r > best_r:
                        best, best_r = n, r
        if best is not None and best_r >= settings.resolve_fuzzy:
            return best

        # 4) embedding similarity (same type only, needs a description to be meaningful)
        if self.embed_fn and description and type:
            ids, mat = self.store.embedding_matrix(world_id)
            if ids:
                same_type = {n.id for n, _ in ents if n.type == type and not numbers_conflict(name, n.name)}
                idx = [i for i, nid in enumerate(ids) if nid in same_type]
                if idx:
                    q = self.embed_fn([f"{type}: {name}. {description}"])[0]
                    sims = mat[idx] @ q
                    j = int(np.argmax(sims))
                    if sims[j] >= settings.resolve_embed:
                        return self.store.get_node(ids[idx[j]])
        return None
