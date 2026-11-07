"""Chronicler: scene -> structured facts -> knowledge-graph writes (PLAN.md §3.1-3.6)."""
from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from .. import prompts as P
from ..config import settings
from ..llm import LLMError
from ..memory import EntityResolver, MemoryStore, Node
from ..memory.resolver import GENERIC, content_tokens
from ..clocks import SEGMENTS_BY_SEVERITY, Clocks, describe
from ..schemas import Extraction, PlayerDelta

log = logging.getLogger("loreweaver.chronicler")

OPEN_STATUSES = ("opened", "advanced", "open")
GONE = ("dead", "destroyed", "deceased", "killed", "slain")
NOISE_STATUS = {"active", "present", "normal", "existing", "unknown", "owned", "here", "current", "ok", "fine",
                "visible", "idle", "stable", "exists", "neutral", "available", "functional", "intact"}
ATTITUDES = {"hostile", "allied", "ally", "friendly", "suspicious", "wary", "loyal", "aggressive", "enemy",
              "distrustful", "trusting", "betrayed", "angry", "grateful", "fearful", "indifferent"}
NOISE_RELATIONS = {"unknown", "none", "neutral", "related", "related_to", "n/a", "na", "null", "no_relation"}


def _clean_relation(rel: str) -> str:
    r = re.sub(r"[^a-z_ ]", "", rel.strip().lower()).replace(" ", "_")
    return re.sub(r"^(is|are|was|were|has_been)_", "", r)


@dataclass
class SceneRecord:
    world_id: int
    story_id: int
    story_node_id: int
    turn: int
    action: str
    dice: str
    damage: int
    narration: str
    segment_id: int
    player_id: int
    hero_name: str
    combat: bool = False


@dataclass
class CommitResult:
    notes: list[str] = field(default_factory=list)
    player: PlayerDelta = field(default_factory=PlayerDelta)
    touched: set[int] = field(default_factory=set)
    thread_closed: bool = False
    consequences_resolved: int = 0
    extraction: Optional[Extraction] = None
    dropped: list[str] = field(default_factory=list)       # ungrounded facts kept out of memory
    event_ids: list[int] = field(default_factory=list)
    clocks_due: list[int] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)    # canon conflicts found while committing


REVIVE_WORDS = re.compile(r"\b(resurrect\w*|reviv\w*|rais\w* from the dead|back from the dead|returns? from death|"
                          r"surviv\w*|faked? (his|her|their) death|not dead|alive after all|undead|ghost|spirit|"
                          r"brought back|risen|rose again)\b", re.I)


class Grounder:
    """Evidence grounding (PLAN.md §8 B2): a fact may enter memory only if the scene text supports it.

    A name counts as mentioned if it, one of its aliases (or the aliases of the node it resolves to), or a
    distinctive token of it ("Silas" for "Silas Crane") occurs in the hero's action or the narration.
    """

    def __init__(self, resolver: EntityResolver, world_id: int, hero: str, text: str):
        self.resolver, self.world_id, self.hero = resolver, world_id, hero
        self.text = " " + re.sub(r"[^a-z0-9' ]+", " ", text.lower()) + " "
        self._cache: dict[str, bool] = {}

    def _in_text(self, name: str) -> bool:
        n = re.sub(r"[^a-z0-9' ]+", " ", name.lower()).strip()
        n = re.sub(r"^(the|a|an) ", "", n)
        if len(n) >= 3 and f" {n} " in self.text:
            return True
        return any(f" {t} " in self.text or f" {t}'s " in self.text
                   for t in content_tokens(name) if len(t) >= 4 and t not in GENERIC)

    def mentioned(self, name: str, aliases: list[str] | None = None) -> bool:
        if not name:
            return False
        key = name.lower()
        if key not in self._cache:
            ok = self.resolver.is_player_ref(name, self.hero) or self._in_text(name) or \
                any(self._in_text(a) for a in aliases or [])
            if not ok:
                node = self.resolver.resolve(self.world_id, name)
                ok = bool(node) and any(self._in_text(x) for x in [node.name, *node.aliases])
            self._cache[key] = ok
        return self._cache[key]

    def overlap(self, sentence: str) -> float:
        words = {w for w in re.findall(r"[a-z]{4,}", sentence.lower())}
        return len([w for w in words if w[:5] in self.text]) / max(1, len(words))

    def filter(self, ex: Extraction) -> list[str]:
        """Drop ungrounded facts in place; return what was dropped (for logs and the inspector)."""
        dropped: list[str] = []
        keep = []
        for ent in ex.entities:
            (keep if self.mentioned(ent.name, ent.aliases) else dropped).append(ent)
        dropped = [f"entity '{e.name}'" for e in dropped]
        ex.entities = keep

        rels = []
        for r in ex.relations:
            if self.mentioned(r.source) and self.mentioned(r.target):
                rels.append(r)
            else:
                dropped.append(f"relation {r.source} -{r.relation}-> {r.target}")
        ex.relations = rels

        events = []
        for ev in ex.events:
            ev.participants = [p for p in ev.participants if self.mentioned(p)]
            if ev.participants or self.overlap(ev.summary) >= 0.5:
                events.append(ev)
            else:
                dropped.append(f"event '{ev.summary[:60]}'")
        ex.events = events

        for d in ex.decisions:
            d.affected = [a for a in d.affected if self.mentioned(a)]
        for t in ex.threads:
            t.related = [x for x in t.related if self.mentioned(x)]
        seeds = []
        for c in ex.combat_starts:
            base = re.sub(r"\s*#?\d+$", "", c.name).strip()
            if self.mentioned(c.name) or self.mentioned(base) or self.overlap(base) >= 0.5:
                seeds.append(c)
            else:
                dropped.append(f"combatant '{c.name}'")
        ex.combat_starts = seeds
        for field_name in ("items_gained", "items_lost", "items_used"):
            items = getattr(ex.player, field_name)
            ok = [i for i in items if self._in_text(i) or self.overlap(i) >= 0.5]
            dropped += [f"{field_name} '{i}'" for i in items if i not in ok]
            setattr(ex.player, field_name, ok)
        return dropped


def _similar(a: str, b: str) -> float:
    ta, tb = set(re.findall(r"[a-z]{3,}", a.lower())), set(re.findall(r"[a-z]{3,}", b.lower()))
    jac = len(ta & tb) / max(1, len(ta | tb))
    return max(jac, difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio())


class Chronicler:
    def __init__(self, llm, store: MemoryStore, resolver: EntityResolver):
        self.llm = llm
        self.store = store
        self.resolver = resolver
        self.clocks = Clocks(store)

    # ------------------------------------------------------------------ extraction
    def extract(self, rec: SceneRecord, known: list[Node], threads: list[Node], clocks: list[Node]) -> Extraction:
        known_txt = "\n".join(f"- {n.name} ({n.type}{', ' + n.status if n.status else ''})" for n in known[:40])
        prompt = P.CHRONICLER_PROMPT.format(
            known=known_txt or "(none yet)",
            threads="\n".join(f"- {t.name}" for t in threads) or "(none)",
            pending="\n".join(f"- {describe(c)}" for c in clocks if c.state.get("status") == "ticking") or "(none)",
            damage=rec.damage, turn=rec.turn, action=rec.action, dice=rec.dice, narration=rec.narration,
            template=P.CHRONICLER_TEMPLATE.replace("HERO", rec.hero_name))
        system = P.CHRONICLER_SYSTEM.format(hero=rec.hero_name)
        try:
            return self.llm.json(system, prompt, Extraction, temperature=settings.extract_temperature,
                                 max_tokens=1800, agent="chronicler", retries=1)
        except LLMError as e:
            log.error("chronicler extraction failed: %s", e)
            return Extraction()

    # ------------------------------------------------------------------ commit to graph
    def commit(self, rec: SceneRecord, ex: Extraction, threads: list[Node], clocks: list[Node],
               inventory: list[str] | None = None, protected: set[int] | frozenset = frozenset()) -> CommitResult:
        """`protected`: combatants of a running encounter; their status is owned by the combat engine."""
        res = CommitResult(extraction=ex)
        S, W, T, sid = self.store, rec.world_id, rec.turn, rec.story_id
        local: dict[str, int] = {}
        scene_text = f"{rec.action}\n{rec.narration}"
        res.dropped = Grounder(self.resolver, W, rec.hero_name, scene_text).filter(ex)
        if res.dropped:
            log.info("grounding dropped %d facts: %s", len(res.dropped), res.dropped)
        if inventory is not None:          # canon guard: the hero can only use what they carry
            have = {i.lower() for i in inventory} | {i.lower() for i in ex.player.items_gained}
            for item in ex.player.items_used:
                if not any(item.lower() in h or h in item.lower() for h in have):
                    res.violations.append(f"The hero used '{item}', which is not in their inventory.")

        def ref(name: str, type_hint: str | None = None, create_proper: bool = False) -> Optional[int]:
            if not name or not name.strip():
                return None
            key = name.strip().lower()
            if self.resolver.is_player_ref(name, rec.hero_name):
                return rec.player_id
            if key in local:
                return local[key]
            node = self.resolver.resolve(W, name, type_hint)
            if node:
                local[key] = node.id
                return node.id
            # unknown proper noun in a relation/event: create it rather than lose the fact
            words = name.split()
            if create_proper and name[:1].isupper() and len(words) <= 5:
                nid = S.add_node(W, sid, type_hint or "character", name.strip(), turn=T, importance=4)
                self._link_new(rec, nid)
                local[key] = nid
                res.notes.append(f"New: {name.strip()}")
                return nid
            return None

        def location_id(name: str | None, create: bool = True) -> Optional[int]:
            if not name:
                return None
            nid = ref(name, "location")
            if nid is not None and S.get_node(nid).type not in ("location", "lore"):
                return None                                 # "Baelen's palm" is a holder, not a place
            owner = re.match(r"^(.+?)['’]s\s", name.strip())
            if nid is None and owner and self.resolver.resolve(W, owner.group(1)):
                return None
            if nid is None and create:
                nid = S.add_node(W, sid, "location", name.strip(), turn=T, importance=4)
                self._link_new(rec, nid)
                local[name.strip().lower()] = nid
                res.notes.append(f"New place: {name.strip()}")
            return nid

        # 1) entities ------------------------------------------------------------------
        for ent in ex.entities:
            if self.resolver.is_player_ref(ent.name, rec.hero_name):
                if ent.location and not ex.player.location:
                    ex.player.location = ent.location
                continue
            attitude = None
            if ent.status and ent.status.strip().lower() in NOISE_STATUS:
                ent.status = None
            elif ent.status and ent.status.strip().lower() in ATTITUDES:   # disposition, not life state
                attitude, ent.status = ent.status.strip().lower(), None
            if ent.type == "location":                     # places are not "located at" the current scene
                ent.location = None
            ent.aliases = [a for a in ent.aliases if not self.resolver.is_player_ref(a, rec.hero_name)
                           and (content_tokens(a) - GENERIC)]
            node = self.resolver.resolve(W, ent.name, ent.type, ent.description, ent.aliases)
            state = {k: v for k, v in (("status", ent.status), ("location", ent.location), ("attitude", attitude)) if v}
            if node and node.id in protected:
                state.pop("status", None)
                ent.status = None
            if node and node.status.lower() in GONE and ent.status and ent.status.lower() not in GONE \
                    and not REVIVE_WORDS.search(scene_text):
                # canon guard: the dead stay dead unless the scene itself justifies a return
                res.violations.append(f"{node.name} is {node.status}, but the scene treats them as {ent.status}.")
                state.pop("status", None)
                ent.status = None
            if node:
                if sid not in S.node_story_ids(node.id):      # returning from a past tale
                    S.upsert_edge(W, sid, node.id, rec.story_node_id, "appears_in", turn=T)
                    res.notes.append(f"Returns from a past tale: {node.name}")
                before = node.status
                desc = node.description
                if ent.description and ent.description.lower() not in desc.lower():
                    desc = ent.description if not desc else (desc if len(desc) > 350 else f"{desc} {ent.description}")
                aliases = list(ent.aliases)
                if ent.name.strip().lower() != node.name.lower():
                    aliases.append(ent.name.strip())
                node = S.update_node(node.id, sid, T, description=desc[:450], state=state,
                                     importance=min(10.0, node.importance + 0.25), aliases_add=aliases)
                if ent.status and before and ent.status.lower() != before.lower():
                    res.notes.append(f"{node.name}: {before} → {ent.status}")
                prev_att = str(node.state.get("attitude", "") or "")
                if attitude and prev_att and attitude != prev_att.lower():
                    res.notes.append(f"{node.name} is now {attitude}")
                nid = node.id
            else:
                nid = S.add_node(W, sid, ent.type, ent.name.strip(), description=ent.description, state=state,
                                 turn=T, aliases=ent.aliases, importance=5)
                self._link_new(rec, nid)
                res.notes.append(f"New {ent.type}: {ent.name.strip()}")
            for k in [ent.name, *ent.aliases]:
                local[k.strip().lower()] = nid
            S.upsert_edge(W, sid, rec.segment_id, nid, "mentions", turn=T)
            res.touched.add(nid)
            if ent.location:
                loc = location_id(ent.location, create=ent.type in ("character", "creature", "faction"))
                if loc and loc != nid:
                    S.replace_edge(W, sid, nid, "located_at", loc, T)

        # 2) relations -----------------------------------------------------------------
        for r in ex.relations:
            rel = _clean_relation(r.relation)
            if not rel or (rel in NOISE_RELATIONS and not r.sentiment):
                continue
            a, b = ref(r.source, create_proper=True), ref(r.target, create_proper=True)
            if a and b and a != b:
                S.upsert_edge(W, sid, a, b, rel, sentiment=r.sentiment, note=r.relation, turn=T)
                res.touched.update((a, b))

        # 3) events --------------------------------------------------------------------
        event_ids: list[int] = []
        for ev in ex.events:
            eid = S.add_node(W, sid, "event", ev.summary[:90], description=ev.summary, importance=ev.importance,
                             turn=T)
            event_ids.append(eid)
            res.event_ids.append(eid)
            S.upsert_edge(W, sid, eid, rec.segment_id, "from_scene", turn=T)
            for p in ev.participants:
                pid = ref(p, create_proper=True)
                if pid:
                    S.upsert_edge(W, sid, eid, pid, "involves", turn=T)
                    node = S.get_node(pid)
                    if node and node.importance < ev.importance - 1:
                        S.update_node(pid, sid, T, importance=float(ev.importance - 1))
                    res.touched.add(pid)
            loc = location_id(ev.location) if ev.location else None
            if loc:
                S.upsert_edge(W, sid, eid, loc, "occurred_at", turn=T)

        # 4) plot threads --------------------------------------------------------------
        for th in ex.threads:
            match = max(threads, key=lambda t: _similar(t.name, th.title), default=None)
            if match and _similar(match.name, th.title) >= 0.55:
                S.update_node(match.id, sid, T, description=th.description or None,
                              state={"status": th.status, f"{th.status}_turn": T})
                tid = match.id
                if th.status in ("resolved", "failed"):
                    res.thread_closed = True
                    res.notes.append(f"Thread {th.status}: {match.name}")
            elif th.status in ("opened", "advanced"):
                tid = S.add_node(W, sid, "thread", th.title[:100], description=th.description,
                                 state={"status": "opened", "opened_turn": T}, importance=7, turn=T)
                S.upsert_edge(W, sid, tid, rec.story_node_id, "appears_in", turn=T)
                threads.append(S.get_node(tid))
                res.notes.append(f"New thread: {th.title}")
            else:
                continue
            for rname in th.related:
                rid = ref(rname)
                if rid:
                    S.upsert_edge(W, sid, tid, rid, "concerns", turn=T)
            for eid in event_ids:
                S.upsert_edge(W, sid, eid, tid, "resolves" if th.status in ("resolved", "failed") else "advances",
                              turn=T)

        # 5) decisions -> future consequences ---------------------------------------------
        hero_tokens = content_tokens(rec.hero_name) | {"you", "hero"}
        past = [n for n in S.nodes(W, ("decision",), story_id=sid)]
        for d in ex.decisions:
            words = set(re.findall(r"[a-z']+", d.decision.lower()))
            if not (words & hero_tokens):                   # only the hero's own choices
                continue
            if any(_similar(p.name, d.decision) >= 0.75 for p in past):
                continue                                    # already remembered
            did = S.add_node(W, sid, "decision", d.decision[:120],
                             state={"status": "pending", "consequence": d.consequence, "severity": d.severity},
                             importance=4 + d.severity, turn=T)
            S.upsert_edge(W, sid, rec.player_id, did, "made", turn=T)
            S.upsert_edge(W, sid, did, rec.segment_id, "made_in", turn=T)
            affected = []
            for a in d.affected:
                aid = ref(a)
                if aid and aid != rec.player_id:
                    S.upsert_edge(W, sid, did, aid, "affects", turn=T)
                    affected.append(aid)
            past.append(S.get_node(did))
            res.notes.append(f"Decision remembered: {d.decision[:80]}")
            if d.consequence:                               # the consequence becomes a ticking clock
                name = d.clock or " ".join(d.consequence.split()[:6])
                same = self.clocks.match(clocks, name, 0.7)
                if same is not None and same.state.get("front"):
                    # a decision's consequence is its own thread, never folded into the campaign front
                    name = " ".join(d.consequence.split()[:7])
                    same = self.clocks.match([c for c in clocks if not c.state.get("front")], name, 0.7)
                if same is None:
                    cid = self.clocks.create(W, sid, T, name, d.consequence, SEGMENTS_BY_SEVERITY[d.severity],
                                             threatens=affected, decision_id=did, story_node_id=rec.story_node_id)
                    clocks.append(S.get_node(cid))
                    res.notes.append(f"⏱ New clock: {name} ({SEGMENTS_BY_SEVERITY[d.severity]} segments)")
                else:                                       # the same threat again: push it, link the cause
                    S.upsert_edge(W, sid, did, same.id, "started", turn=T)
                    res.notes += self.clocks.tick(same.id, 1, sid, T, d.decision, source="decision")[1]

        # 6) clocks: ticks pushed by the scene, threats the hero neutralized -------------------
        ticking = [c for c in clocks if c.state.get("status") == "ticking"]
        ticked: set[int] = set()
        for t in ex.clock_ticks:
            c = self.clocks.match(ticking, t.clock)
            if c and c.id not in ticked and c.turn < T:     # a clock never ticks in the scene that created it
                ticked.add(c.id)
                node, notes = self.clocks.tick(c.id, t.ticks, sid, T, t.reason)
                res.notes += notes
                if node and node.state.get("status") == "due":
                    res.clocks_due.append(node.id)
        narration_norm = re.sub(r"\s+", " ", rec.narration.lower())
        for a in ex.clocks_averted:
            c = self.clocks.match(ticking, a.clock)
            quote = re.sub(r"\s+", " ", a.quote.lower()).strip(" .\"'")
            if c and len(quote) >= 15 and quote in narration_norm:  # must be evidenced by the scene itself
                res.notes += self.clocks.avert(c.id, sid, T, a.quote)

        # 7) hero inventory & location ---------------------------------------------------
        pdelta = ex.player
        for item in pdelta.items_gained:
            iid = ref(item, "item")
            if iid is None:
                iid = S.add_node(W, sid, "item", item.strip(), turn=T, importance=5)
                self._link_new(rec, iid)
            S.upsert_edge(W, sid, rec.player_id, iid, "owns", turn=T)
        for item in pdelta.items_lost:
            iid = ref(item, "item")
            if iid:
                S.delete_edge(rec.player_id, iid, "owns")
                S.upsert_edge(W, sid, rec.player_id, iid, "once_owned", turn=T)
        if pdelta.location:
            loc = location_id(pdelta.location)
            if loc:
                S.replace_edge(W, sid, rec.player_id, "located_at", loc, T)
                pdelta.location = S.get_node(loc).name
        res.player = pdelta
        for nid in res.touched:
            S.touch(nid, sid, T)
        return res

    def _link_new(self, rec: SceneRecord, nid: int):
        self.store.upsert_edge(rec.world_id, rec.story_id, nid, rec.story_node_id, "appears_in", turn=rec.turn)
        self.store.upsert_edge(rec.world_id, rec.story_id, rec.segment_id, nid, "mentions", turn=rec.turn)
