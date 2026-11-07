"""SQLite persistence for the story knowledge graph.

Everything (entities, events, decisions, threads, scenes, summaries) is a row in `nodes`;
everything that connects them is a row in `edges`. Worlds own nodes, so stories in the same world
share one graph. State changes are versioned in `history` (temporal memory). `fts` gives BM25
keyword search and `embeddings` stores normalized vectors for dense search.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

ENTITY_NODE_TYPES = ("player", "character", "creature", "location", "item", "faction", "lore")
NARRATIVE_NODE_TYPES = ("event", "thread", "decision", "segment", "summary")

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS worlds(
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, theme TEXT, bible TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS stories(
  id INTEGER PRIMARY KEY, world_id INTEGER NOT NULL REFERENCES worlds(id) ON DELETE CASCADE,
  title TEXT, theme TEXT, turn INTEGER DEFAULT 0, status TEXT DEFAULT 'active',
  player TEXT, arc TEXT, meta TEXT, created_at REAL, updated_at REAL);
CREATE TABLE IF NOT EXISTS nodes(
  id INTEGER PRIMARY KEY, world_id INTEGER NOT NULL REFERENCES worlds(id) ON DELETE CASCADE,
  story_id INTEGER, type TEXT NOT NULL, name TEXT NOT NULL, aliases TEXT DEFAULT '[]',
  description TEXT DEFAULT '', content TEXT DEFAULT '', state TEXT DEFAULT '{}',
  importance REAL DEFAULT 5, turn INTEGER DEFAULT 0, updated_turn INTEGER DEFAULT 0,
  last_seen_turn INTEGER DEFAULT 0, last_seen_story INTEGER, created_at REAL);
CREATE INDEX IF NOT EXISTS ix_nodes_world_type ON nodes(world_id, type);
CREATE INDEX IF NOT EXISTS ix_nodes_story ON nodes(story_id);
CREATE TABLE IF NOT EXISTS node_stories(
  node_id INTEGER REFERENCES nodes(id) ON DELETE CASCADE, story_id INTEGER,
  first_turn INTEGER, last_turn INTEGER, PRIMARY KEY(node_id, story_id));
CREATE TABLE IF NOT EXISTS edges(
  id INTEGER PRIMARY KEY, world_id INTEGER NOT NULL, story_id INTEGER,
  src INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
  dst INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
  rel TEXT NOT NULL, weight REAL DEFAULT 1, sentiment INTEGER DEFAULT 0, note TEXT DEFAULT '',
  turn INTEGER DEFAULT 0, updated_turn INTEGER DEFAULT 0, UNIQUE(src, dst, rel));
CREATE INDEX IF NOT EXISTS ix_edges_world ON edges(world_id);
CREATE TABLE IF NOT EXISTS history(
  id INTEGER PRIMARY KEY, node_id INTEGER REFERENCES nodes(id) ON DELETE CASCADE,
  story_id INTEGER, turn INTEGER, field TEXT, old TEXT, new TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS embeddings(
  node_id INTEGER PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE, vec BLOB, text_hash TEXT);
CREATE TABLE IF NOT EXISTS turns(
  id INTEGER PRIMARY KEY, story_id INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
  turn INTEGER, action TEXT, check_json TEXT, narration TEXT, choices TEXT,
  retrieval TEXT, notes TEXT, segment_id INTEGER, created_at REAL, UNIQUE(story_id, turn));
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
  name, body, world_id UNINDEXED, tokenize='porter unicode61');
"""


def _j(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False)


@dataclass
class Node:
    id: int
    world_id: int
    story_id: Optional[int]
    type: str
    name: str
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    content: str = ""
    state: dict = field(default_factory=dict)
    importance: float = 5.0
    turn: int = 0
    updated_turn: int = 0
    last_seen_turn: int = 0
    last_seen_story: Optional[int] = None

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> "Node":
        return cls(id=r["id"], world_id=r["world_id"], story_id=r["story_id"], type=r["type"],
                   name=r["name"], aliases=json.loads(r["aliases"] or "[]"),
                   description=r["description"] or "", content=r["content"] or "",
                   state=json.loads(r["state"] or "{}"), importance=r["importance"], turn=r["turn"],
                   updated_turn=r["updated_turn"], last_seen_turn=r["last_seen_turn"],
                   last_seen_story=r["last_seen_story"])

    @property
    def status(self) -> str:
        return str(self.state.get("status", "") or "")

    @property
    def is_entity(self) -> bool:
        return self.type in ENTITY_NODE_TYPES

    def text(self) -> str:
        """Canonical text used for embeddings and keyword search."""
        st = ", ".join(f"{k}: {v}" for k, v in self.state.items() if v not in (None, "", [])
                       and k not in ("consequence",))
        if self.type in ("segment", "summary"):
            return self.content or self.description
        if self.type == "event":
            return f"Event (turn {self.turn}): {self.description or self.name}"
        if self.type == "thread":
            return f"Plot thread: {self.name}. {self.description} ({st})"
        if self.type == "decision":
            return (f"Hero's decision: {self.name}. Possible consequence: "
                    f"{self.state.get('consequence', '')} ({self.state.get('status', '')})")
        alias = f" (aka {', '.join(self.aliases)})" if self.aliases else ""
        return f"{self.type}: {self.name}{alias}. {self.description}" + (f" [{st}]" if st else "")


@dataclass
class Edge:
    id: int
    src: int
    dst: int
    rel: str
    weight: float
    sentiment: int
    note: str
    story_id: Optional[int]
    turn: int
    updated_turn: int
    valid_to: Optional[int] = None     # turn at which a later, contradicting fact superseded this one

    @property
    def current(self) -> bool:
        return self.valid_to is None

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> "Edge":
        return cls(id=r["id"], src=r["src"], dst=r["dst"], rel=r["rel"], weight=r["weight"],
                   sentiment=r["sentiment"], note=r["note"] or "", story_id=r["story_id"],
                   turn=r["turn"], updated_turn=r["updated_turn"],
                   valid_to=r["valid_to"] if "valid_to" in r.keys() else None)


class MemoryStore:
    def __init__(self, path: str | Path = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.version = 0          # bumps on any graph/embedding write -> invalidates caches
        with self.lock:
            self.conn.executescript(SCHEMA)
            cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(edges)")}
            if "valid_to" not in cols:     # migration: temporal validity of relations (Zep-style invalidation)
                self.conn.execute("ALTER TABLE edges ADD COLUMN valid_to INTEGER")
            self.conn.commit()

    # ------------------------------------------------------------------ plumbing
    def _exec(self, sql: str, params: Iterable = ()) -> sqlite3.Cursor:
        with self.lock:
            cur = self.conn.execute(sql, tuple(params))
            self.conn.commit()
            return cur

    def _all(self, sql: str, params: Iterable = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, tuple(params)).fetchall()

    def _one(self, sql: str, params: Iterable = ()) -> Optional[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, tuple(params)).fetchone()

    def _bump(self):
        self.version += 1

    def close(self):
        with self.lock:
            self.conn.close()

    # ------------------------------------------------------------------ worlds & stories
    def create_world(self, name: str, theme: str, bible: dict) -> int:
        return self._exec("INSERT INTO worlds(name, theme, bible, created_at) VALUES(?,?,?,?)",
                          (name, theme, _j(bible), time.time())).lastrowid

    def get_world(self, world_id: int) -> dict:
        r = self._one("SELECT * FROM worlds WHERE id=?", (world_id,))
        return {**dict(r), "bible": json.loads(r["bible"] or "{}")} if r else {}

    def list_worlds(self) -> list[dict]:
        rows = self._all("""SELECT w.*, COUNT(DISTINCT s.id) AS n_stories,
                            (SELECT COUNT(*) FROM nodes n WHERE n.world_id=w.id) AS n_nodes
                            FROM worlds w LEFT JOIN stories s ON s.world_id=w.id
                            GROUP BY w.id ORDER BY w.created_at DESC""")
        return [dict(r) for r in rows]

    def update_world_bible(self, world_id: int, bible: dict):
        self._exec("UPDATE worlds SET bible=? WHERE id=?", (_j(bible), world_id))

    def create_story(self, world_id: int, title: str, theme: str, player: dict,
                     arc: dict | None = None, meta: dict | None = None) -> int:
        now = time.time()
        return self._exec("""INSERT INTO stories(world_id, title, theme, player, arc, meta, created_at, updated_at)
                             VALUES(?,?,?,?,?,?,?,?)""",
                          (world_id, title, theme, _j(player), _j(arc or {}), _j(meta or {}), now, now)).lastrowid

    def get_story(self, story_id: int) -> dict:
        r = self._one("SELECT * FROM stories WHERE id=?", (story_id,))
        if not r:
            return {}
        d = dict(r)
        for k in ("player", "arc", "meta"):
            d[k] = json.loads(d[k] or "{}")
        return d

    def update_story(self, story_id: int, **fields):
        if not fields:
            return
        sets, vals = [], []
        for k, v in fields.items():
            sets.append(f"{k}=?")
            vals.append(_j(v) if k in ("player", "arc", "meta") else v)
        sets.append("updated_at=?")
        vals.append(time.time())
        self._exec(f"UPDATE stories SET {', '.join(sets)} WHERE id=?", (*vals, story_id))

    def list_stories(self, world_id: int | None = None) -> list[dict]:
        sql = """SELECT s.id, s.world_id, s.title, s.theme, s.turn, s.status, s.updated_at, w.name AS world_name
                 FROM stories s JOIN worlds w ON w.id=s.world_id"""
        rows = self._all(sql + (" WHERE s.world_id=?" if world_id else "") + " ORDER BY s.updated_at DESC",
                         (world_id,) if world_id else ())
        return [dict(r) for r in rows]

    def delete_story(self, story_id: int):
        with self.lock:
            ids = [r["id"] for r in self.conn.execute(
                "SELECT id FROM nodes WHERE story_id=? AND type IN ('segment','summary','event','decision','thread')",
                (story_id,))]
            for nid in ids:
                self.conn.execute("DELETE FROM fts WHERE rowid=?", (nid,))
            self.conn.execute("DELETE FROM nodes WHERE story_id=? AND type IN "
                              "('segment','summary','event','decision','thread')", (story_id,))
            self.conn.execute("DELETE FROM stories WHERE id=?", (story_id,))
            self.conn.commit()
        self._bump()

    # ------------------------------------------------------------------ nodes
    def add_node(self, world_id: int, story_id: Optional[int], type: str, name: str, *,
                 description: str = "", content: str = "", state: dict | None = None,
                 importance: float = 5.0, turn: int = 0, aliases: list[str] | None = None) -> int:
        with self.lock:
            nid = self.conn.execute(
                """INSERT INTO nodes(world_id, story_id, type, name, aliases, description, content, state,
                   importance, turn, updated_turn, last_seen_turn, last_seen_story, created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (world_id, story_id, type, name.strip(), _j(aliases or []), description, content,
                 _j(state or {}), importance, turn, turn, turn, story_id, time.time())).lastrowid
            if story_id is not None:
                self.conn.execute("INSERT OR IGNORE INTO node_stories VALUES(?,?,?,?)", (nid, story_id, turn, turn))
            self.conn.commit()
            self._index_fts(self.get_node(nid))
        self._bump()
        return nid

    def _index_fts(self, node: Node):
        with self.lock:
            self.conn.execute("DELETE FROM fts WHERE rowid=?", (node.id,))
            self.conn.execute("INSERT INTO fts(rowid, name, body, world_id) VALUES(?,?,?,?)",
                              (node.id, " ".join([node.name, *node.aliases]), node.text(), node.world_id))
            self.conn.commit()

    def get_node(self, node_id: int) -> Optional[Node]:
        r = self._one("SELECT * FROM nodes WHERE id=?", (node_id,))
        return Node.from_row(r) if r else None

    def get_nodes(self, ids: Iterable[int]) -> dict[int, Node]:
        ids = list(dict.fromkeys(ids))
        out: dict[int, Node] = {}
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            for r in self._all(f"SELECT * FROM nodes WHERE id IN ({','.join('?' * len(chunk))})", chunk):
                out[r["id"]] = Node.from_row(r)
        return out

    def nodes(self, world_id: int, types: Iterable[str] | None = None,
              story_id: int | None = None) -> list[Node]:
        sql, p = "SELECT * FROM nodes WHERE world_id=?", [world_id]
        if types:
            types = list(types)
            sql += f" AND type IN ({','.join('?' * len(types))})"
            p += types
        if story_id is not None:
            sql += " AND story_id=?"
            p.append(story_id)
        return [Node.from_row(r) for r in self._all(sql + " ORDER BY id", p)]

    def update_node(self, node_id: int, story_id: Optional[int], turn: int, *,
                    description: str | None = None, content: str | None = None,
                    state: dict | None = None, importance: float | None = None,
                    aliases_add: Iterable[str] = (), name: str | None = None) -> Node:
        """Merge updates into a node; state changes are recorded in `history`."""
        with self.lock:
            node = self.get_node(node_id)
            if node is None:
                raise KeyError(node_id)
            now = time.time()
            if state:
                for k, v in state.items():
                    if v in (None, ""):
                        continue
                    old = node.state.get(k)
                    if str(old) != str(v):
                        self.conn.execute("INSERT INTO history(node_id, story_id, turn, field, old, new, created_at)"
                                          " VALUES(?,?,?,?,?,?,?)",
                                          (node_id, story_id, turn, k, None if old is None else str(old), str(v), now))
                        node.state[k] = v
            if description and description.strip() and description.strip() != node.description:
                node.description = description.strip()
            if content is not None:
                node.content = content
            if importance is not None:
                node.importance = max(node.importance, importance)
            if name:
                node.name = name
            low = {a.lower() for a in node.aliases} | {node.name.lower()}
            for a in aliases_add:
                a = a.strip()
                if a and a.lower() not in low:
                    node.aliases.append(a)
                    low.add(a.lower())
            self.conn.execute(
                """UPDATE nodes SET name=?, aliases=?, description=?, content=?, state=?, importance=?,
                   updated_turn=?, last_seen_turn=?, last_seen_story=? WHERE id=?""",
                (node.name, _j(node.aliases), node.description, node.content, _j(node.state), node.importance,
                 turn, turn, story_id if story_id is not None else node.last_seen_story, node_id))
            self.touch(node_id, story_id, turn, commit=False)
            self.conn.commit()
            node = self.get_node(node_id)
            self._index_fts(node)
        self._bump()
        return node

    def touch(self, node_id: int, story_id: Optional[int], turn: int, commit: bool = True):
        """Mark a node as seen in a story at a turn (drives recency + cross-story links)."""
        with self.lock:
            self.conn.execute("UPDATE nodes SET last_seen_turn=?, last_seen_story=? WHERE id=?",
                              (turn, story_id, node_id))
            if story_id is not None:
                self.conn.execute("""INSERT INTO node_stories VALUES(?,?,?,?)
                                     ON CONFLICT(node_id, story_id) DO UPDATE SET last_turn=excluded.last_turn""",
                                  (node_id, story_id, turn, turn))
            if commit:
                self.conn.commit()

    def node_story_ids(self, node_id: int) -> list[int]:
        return [r["story_id"] for r in self._all("SELECT story_id FROM node_stories WHERE node_id=?", (node_id,))]

    def history(self, node_id: int) -> list[dict]:
        return [dict(r) for r in self._all("SELECT * FROM history WHERE node_id=? ORDER BY id", (node_id,))]

    def world_history(self, world_id: int, limit: int = 200) -> list[dict]:
        rows = self._all("""SELECT h.*, n.name, n.type FROM history h JOIN nodes n ON n.id=h.node_id
                            WHERE n.world_id=? ORDER BY h.id DESC LIMIT ?""", (world_id, limit))
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ edges
    def upsert_edge(self, world_id: int, story_id: Optional[int], src: int, dst: int, rel: str, *,
                    weight: float = 1.0, sentiment: int = 0, note: str = "", turn: int = 0) -> int:
        if src == dst:
            return -1
        rel = rel.strip().lower().replace(" ", "_")[:40] or "related_to"
        with self.lock:
            if abs(sentiment) >= 3:
                # Temporal invalidation (Zep/Graphiti): a strong new feeling of src toward dst that flips the
                # sign of an older relation supersedes it ("ally_of" +6, later "betrayed" -8). The old edge is
                # kept for history but marked valid_to=turn, so current-state views no longer show it.
                self.conn.execute("""UPDATE edges SET valid_to=? WHERE src=? AND dst=? AND rel<>? AND valid_to IS NULL
                                     AND sentiment * ? < 0 AND ABS(sentiment) >= 3""",
                                  (turn, src, dst, rel, sentiment))
            r = self.conn.execute("SELECT * FROM edges WHERE src=? AND dst=? AND rel=?", (src, dst, rel)).fetchone()
            if r:
                if sentiment and r["sentiment"] * sentiment < 0 and abs(r["sentiment"]) >= 3 and abs(sentiment) >= 3:
                    new_sent = sentiment                 # a reversal replaces the old value instead of averaging
                else:
                    new_sent = r["sentiment"] if not sentiment else round(0.4 * r["sentiment"] + 0.6 * sentiment)
                self.conn.execute("""UPDATE edges SET weight=?, sentiment=?, note=?, updated_turn=?, valid_to=NULL,
                                     story_id=COALESCE(?, story_id) WHERE id=?""",
                                  (min(r["weight"] + 0.5 * weight, 10), new_sent, note or r["note"], turn,
                                   story_id, r["id"]))
                eid = r["id"]
            else:
                eid = self.conn.execute(
                    """INSERT INTO edges(world_id, story_id, src, dst, rel, weight, sentiment, note, turn, updated_turn)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (world_id, story_id, src, dst, rel, weight, sentiment, note, turn, turn)).lastrowid
            self.conn.commit()
        self._bump()
        return eid

    def delete_edge(self, src: int, dst: int, rel: str):
        self._exec("DELETE FROM edges WHERE src=? AND dst=? AND rel=?", (src, dst, rel))
        self._bump()

    def replace_edge(self, world_id: int, story_id: Optional[int], src: int, rel: str, dst: int, turn: int):
        """Single-valued relation (e.g. located_at): drop the old target, keep the new one."""
        with self.lock:
            self.conn.execute("DELETE FROM edges WHERE src=? AND rel=? AND dst<>?", (src, rel, dst))
            self.conn.commit()
        self.upsert_edge(world_id, story_id, src, dst, rel, turn=turn)

    def edges(self, world_id: int) -> list[Edge]:
        return [Edge.from_row(r) for r in self._all("SELECT * FROM edges WHERE world_id=?", (world_id,))]

    def edges_of(self, node_id: int) -> list[Edge]:
        return [Edge.from_row(r) for r in
                self._all("SELECT * FROM edges WHERE src=? OR dst=? ORDER BY updated_turn DESC", (node_id, node_id))]

    # ------------------------------------------------------------------ embeddings
    @staticmethod
    def text_hash(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    def stale_embeddings(self, world_id: int) -> list[Node]:
        nodes = self.nodes(world_id)
        hashes = {r["node_id"]: r["text_hash"] for r in self._all(
            "SELECT e.node_id, e.text_hash FROM embeddings e JOIN nodes n ON n.id=e.node_id WHERE n.world_id=?",
            (world_id,))}
        return [n for n in nodes if hashes.get(n.id) != self.text_hash(n.text())]

    def set_embeddings(self, items: list[tuple[int, np.ndarray, str]]):
        if not items:
            return
        with self.lock:
            self.conn.executemany("INSERT OR REPLACE INTO embeddings(node_id, vec, text_hash) VALUES(?,?,?)",
                                  [(nid, vec.astype(np.float32).tobytes(), h) for nid, vec, h in items])
            self.conn.commit()
        self._bump()

    def embedding_matrix(self, world_id: int) -> tuple[list[int], np.ndarray]:
        rows = self._all("SELECT e.node_id, e.vec FROM embeddings e JOIN nodes n ON n.id=e.node_id "
                         "WHERE n.world_id=?", (world_id,))
        if not rows:
            return [], np.zeros((0, 1), dtype=np.float32)
        return [r["node_id"] for r in rows], np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])

    # ------------------------------------------------------------------ keyword search
    def fts_search(self, world_id: int, query: str, limit: int = 20) -> list[tuple[int, float]]:
        import re
        terms = [t for t in re.findall(r"[A-Za-z0-9']{3,}", query.lower()) if t not in _STOP]
        if not terms:
            return []
        q = " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))
        try:
            rows = self._all("SELECT rowid, bm25(fts, 3.0, 1.0) AS s FROM fts WHERE fts MATCH ? AND world_id=? "
                             "ORDER BY s LIMIT ?", (q, world_id, limit))
        except sqlite3.OperationalError:
            return []
        return [(r["rowid"], -r["s"]) for r in rows]

    # ------------------------------------------------------------------ turns
    def save_turn(self, story_id: int, turn: int, *, action: str, check: dict | None, narration: str,
                  choices: list[dict], retrieval: dict | None = None, notes: list[str] | None = None,
                  segment_id: int | None = None):
        self._exec("""INSERT OR REPLACE INTO turns(story_id, turn, action, check_json, narration, choices,
                      retrieval, notes, segment_id, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                   (story_id, turn, action, _j(check), narration, _j(choices), _j(retrieval or {}),
                    _j(notes or []), segment_id, time.time()))

    def update_turn_choices(self, story_id: int, turn: int, choices: list[dict]):
        self._exec("UPDATE turns SET choices=? WHERE story_id=? AND turn=?", (_j(choices), story_id, turn))

    def update_turn_narration(self, story_id: int, turn: int, narration: str):
        self._exec("UPDATE turns SET narration=? WHERE story_id=? AND turn=?", (narration, story_id, turn))

    def update_turn_notes(self, story_id: int, turn: int, notes: list[str]):
        self._exec("UPDATE turns SET notes=? WHERE story_id=? AND turn=?", (_j(notes), story_id, turn))

    def turns(self, story_id: int, last: int | None = None) -> list[dict]:
        if last:
            rows = self._all("SELECT * FROM (SELECT * FROM turns WHERE story_id=? ORDER BY turn DESC LIMIT ?) "
                             "ORDER BY turn", (story_id, last))
        else:
            rows = self._all("SELECT * FROM turns WHERE story_id=? ORDER BY turn", (story_id,))
        out = []
        for r in rows:
            d = dict(r)
            for k, default in (("check_json", None), ("choices", []), ("retrieval", {}), ("notes", [])):
                d[k] = json.loads(d[k]) if d[k] else default
            out.append(d)
        return out

    # ------------------------------------------------------------------ stats
    def stats(self, world_id: int) -> dict:
        rows = self._all("SELECT type, COUNT(*) c FROM nodes WHERE world_id=? GROUP BY type", (world_id,))
        e = self._one("SELECT COUNT(*) c FROM edges WHERE world_id=?", (world_id,))
        return {"nodes": {r["type"]: r["c"] for r in rows}, "edges": e["c"] if e else 0}


_STOP = set("""the and for with that this from you your are was were his her their them they she him
have has had not but into onto out over what when where which who whom will would could should can
about there then than its it's our ours all any some just like very more most also been being
does did doing each few other such only own same too here off again once under until while""".split())
