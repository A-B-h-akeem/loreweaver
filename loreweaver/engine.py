"""GameEngine: orchestrates agents, rules and memory for one story at a time (PLAN.md §4)."""
from __future__ import annotations

import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

import numpy as np

from .agents import (Adjudicator, Chronicler, CommitResult, Director, Narrator, SceneContext, SceneRecord,
                     StreamFilter, Summarizer, Worldbuilder, bible_text)
from .agents.continuity import ContinuityChecker
from .agents.narrator import scrub
from .clocks import Clocks, describe
from .combat import Encounter, budget_seeds, classify, combat_choices, make_enemy, resolve_round, wants_combat
from .config import settings
from .llm import OllamaLLM
from .memory import ENTITY_NODE_TYPES, EntityResolver, MemoryRetriever, MemoryStore, Node, RecallResult, StoryGraph
from .rules import Character, CheckResult, roll_check, split_narration
from .schemas import ArcPlan, Choice, WorldBible

log = logging.getLogger("loreweaver.engine")

OPENING_ACTION = "(The adventure begins)"
GONE_STATUSES = ("dead", "destroyed", "deceased", "killed", "slain", "missing", "lost")
OPEN_THREAD = ("opened", "advanced", "open")
_POSSESSIVE = re.compile(r"\bmy\s+((?:[\w'-]+\s+){0,3}?[\w'-]+(?:\s+of\s+(?:the\s+)?[\w'-]+(?:\s+[\w'-]+)?)?)"
                         r"(?=\s+(?:and|to|at|on|against|into|toward|towards|from|with|before|while|so)\b|[,.!?;]|$)",
                         re.I)
# Whole-word item-use verbs right after "you" (a first version matched "use" in "uselessly" and fired on 37%
# of real scenes; the gate exists to skip the checker when nothing risky happens).
_USE_VERBS = re.compile(r"\byou\s+(?:\w+ly\s+)?(raise[sd]?|raising|draw[sn]?|drew|drink[s]?|drank|wield(?:s|ed)?|"
                        r"swing[s]?|swung|throw[sn]?|threw|unleash(?:es|ed)?|use[sd]?|using|cast[s]?|"
                        r"brandish(?:es|ed)?|uncork(?:s|ed)?|activate[sd]?|aim(?:s|ed)?|toss(?:es|ed)?|"
                        r"pull(?:s|ed)? out|hold(?:s)? up|held up|light(?:s|ed)? (?:the|a|your))\b", re.I)
_NOT_ITEMS = {"hand", "hands", "fist", "fists", "voice", "eyes", "eye", "head", "heart", "breath", "mind", "feet",
              "foot", "leg", "legs", "arm", "arms", "back", "name", "friend", "friends", "way", "best", "life",
              "ground", "side", "shoulder", "shoulders", "teeth", "fingers", "knees", "knee", "body", "face", "lips",
              "throat", "ears", "own", "self", "word", "words", "promise", "story", "skin", "blood", "tongue",
              "family", "home", "people", "crew", "ship", "horse", "companion", "companions", "past", "fate",
              "magic", "spell", "spells", "strength", "courage", "senses", "breath", "stance", "guard", "footing",
              "grip", "weight", "turn", "chance", "time", "place", "position", "attention", "thoughts", "memory"}
FALLBACK_CHOICES = [Choice(text="Press forward boldly"), Choice(text="Look around carefully"),
                    Choice(text="Talk to whoever is nearby")]


@dataclass
class TurnPrep:
    turn: int
    action: str
    ctx: SceneContext
    recall: Optional[RecallResult]
    check: Optional[CheckResult] = None
    notes: list[str] = field(default_factory=list)
    firing: list[int] = field(default_factory=list)      # due clocks whose payoff this scene must deliver
    combat: bool = False                                 # this turn was an encounter round


@dataclass
class TurnOutcome:
    turn: int
    action: str
    prose: str
    choices: list[Choice]
    check: Optional[CheckResult]
    notes: list[str]
    segment_id: int


class GameEngine:
    def __init__(self, store: MemoryStore | None = None, llm=None, seed: int | None = None):
        self.store = store or MemoryStore(settings.db_path)
        self.llm = llm or OllamaLLM()
        self.graph = StoryGraph(self.store)
        self.resolver = EntityResolver(self.store, self._embed)
        self.retriever = MemoryRetriever(self.store, self.graph, self._embed)
        self.worldbuilder = Worldbuilder(self.llm)
        self.adjudicator = Adjudicator(self.llm)
        self.narrator = Narrator(self.llm)
        self.chronicler = Chronicler(self.llm, self.store, self.resolver)
        self.director = Director(self.llm)
        self.summarizer = Summarizer(self.llm)
        self.clocks = Clocks(self.store)
        self.continuity = ContinuityChecker(self.llm)
        self.continuity_log: list[dict] = []
        self.rng = random.Random(seed)

        self.world_id: Optional[int] = None
        self.story_id: Optional[int] = None
        self.story_node_id: Optional[int] = None
        self.player_id: Optional[int] = None
        self.character: Optional[Character] = None
        self.arc: Optional[ArcPlan] = None
        self.bible: dict = {}
        self.turn: int = -1
        self._last_plan_turn: int = 0
        self.downed_pending = False
        self.encounter: Optional[Encounter] = None
        self._last_downed_turn: int = -99

        self._bg: Optional[threading.Thread] = None
        self.memory_status = "idle"
        self.memory_error: Optional[str] = None
        self.last_commit: Optional[CommitResult] = None
        self.last_recall: Optional[RecallResult] = None
        self.last_outcome: Optional[TurnOutcome] = None

    # ================================================================== utilities
    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.llm.embed(texts)

    def embed_stale(self):
        stale = self.store.stale_embeddings(self.world_id)
        if not stale:
            return 0
        texts = [n.text() for n in stale]
        try:
            vecs = self._embed(texts)
        except Exception as e:  # noqa: BLE001
            log.error("embedding failed: %s", e)
            return 0
        self.store.set_embeddings([(n.id, vecs[i], self.store.text_hash(texts[i])) for i, n in enumerate(stale)])
        return len(stale)

    def _save_player(self):
        self.store.update_story(self.story_id, player=self.character.model_dump(), turn=max(self.turn, 0))
        self.store.update_node(self.player_id, self.story_id, max(self.turn, 0),
                               state={"location": self.character.location, "level": self.character.level,
                                      "status": "down" if self.character.is_down else "alive"})

    @property
    def busy(self) -> bool:
        return self._bg is not None and self._bg.is_alive()

    def wait_ready(self, timeout: float | None = None) -> bool:
        if self._bg is not None:
            self._bg.join(timeout)
            return not self._bg.is_alive()
        return True

    # ================================================================== story setup
    def new_story(self, theme: str, hero_name: str, hero_class: str, world_id: int | None = None,
                  progress: Callable[[str], None] = lambda m: None) -> int:
        self.wait_ready()
        existing = self._existing_world_text(world_id) if world_id else None
        progress("Worldbuilder is shaping the world…")
        bible = self.worldbuilder.build(theme, hero_name, hero_class, existing)

        if world_id is None:
            world_id = self.store.create_world(bible.world_name, theme, bible.model_dump())
        else:
            self._merge_world_bible(world_id, bible)
        self.world_id = world_id
        self.character = Character.create(hero_name, hero_class)
        self.character.location = bible.starting_location
        self.bible = bible.model_dump()
        self.story_id = self.store.create_story(world_id, bible.title, theme, self.character.model_dump())
        self.turn = 0
        self.encounter = None

        progress("Seeding the memory graph…")
        self._seed_graph(bible)
        if bible.front:                       # the campaign threat, ticking in the background (a front)
            self.clocks.create(world_id, self.story_id, 0, bible.front.name, bible.front.doom or
                               bible.central_conflict, 8, portents=bible.front.portents, front=True,
                               story_node_id=self.story_node_id)
        self.store.update_story(self.story_id, meta={"bible": self.bible, "story_node_id": self.story_node_id,
                                                     "player_id": self.player_id})
        self.embed_stale()

        progress("Director is plotting the arc…")
        self.arc = self.director.plan(world=bible_text(self.bible, include_secrets=True),
                                      sheet=self.character.sheet_text(), turn=0, story_so_far="", recent="",
                                      threads=[], pending=self.pending_consequences(),
                                      dormant=self.dormant_entities(), previous=None)
        self.store.update_story(self.story_id, arc=self.arc.model_dump())
        return self.story_id

    def _existing_world_text(self, world_id: int) -> str:
        world = self.store.get_world(world_id)
        ents = [n for n in self.store.nodes(world_id, types=("character", "creature", "faction", "location", "lore",
                                                             "player"))]
        ents.sort(key=lambda n: -n.importance)
        lines = [bible_text(world.get("bible", {})), "Notable entities:"]
        for n in ents[:18]:
            label = "hero of a past tale" if n.type == "player" else n.type
            lines.append(f"- {n.name} ({label}{', ' + n.status if n.status else ''}): {n.description[:120]}")
        sums = [n for n in self.store.nodes(world_id, types=("summary",)) if n.state.get("level") == 2] or \
               self.store.nodes(world_id, types=("summary",))[-3:]
        if sums:
            lines.append("History of past tales: " + " ".join(s.content[:500] for s in sums[-3:]))
        return "\n".join(lines)

    def _merge_world_bible(self, world_id: int, bible: WorldBible):
        world = self.store.get_world(world_id)
        wb = world.get("bible", {})
        for key in ("locations", "factions", "npcs"):
            names = {x["name"].lower() for x in wb.get(key, [])}
            wb.setdefault(key, []).extend(x.model_dump() for x in getattr(bible, key) if x.name.lower() not in names)
        self.store.update_world_bible(world_id, wb)

    def _get_or_create(self, type_: str, name: str, description: str = "", state: dict | None = None,
                       importance: float = 5.0) -> int:
        node = self.resolver.resolve(self.world_id, name, type_)
        if node:
            if self.story_id not in self.store.node_story_ids(node.id):
                self.store.upsert_edge(self.world_id, self.story_id, node.id, self.story_node_id, "appears_in")
            if state and node.status.lower() in GONE_STATUSES:     # never resurrect via re-seeding
                state = {k: v for k, v in state.items() if k != "status"}
            self.store.update_node(node.id, self.story_id, 0, description=None if node.description else description,
                                   state=state)
            return node.id
        nid = self.store.add_node(self.world_id, self.story_id, type_, name, description=description,
                                  state=state or {}, importance=importance)
        self.store.upsert_edge(self.world_id, self.story_id, nid, self.story_node_id, "appears_in")
        return nid

    def _seed_graph(self, bible: WorldBible):
        S, W, sid = self.store, self.world_id, self.story_id
        self.story_node_id = S.add_node(W, sid, "story", bible.title,
                                        description=f"{bible.central_conflict} (theme: {S.get_story(sid)['theme']})",
                                        importance=8)
        c = self.character
        self.player_id = S.add_node(W, sid, "player", c.name, description=f"The hero of '{bible.title}', a {c.char_class}.",
                                    state={"class": c.char_class, "level": c.level, "status": "alive"}, importance=9)
        S.upsert_edge(W, sid, self.player_id, self.story_node_id, "appears_in")
        loc_ids = {l.name.lower(): self._get_or_create("location", l.name, l.description) for l in bible.locations}
        fac_ids = {f.name.lower(): self._get_or_create("faction", f.name, f"{f.description} Goal: {f.goal}".strip(),
                                                       importance=6) for f in bible.factions}
        conflict = S.add_node(W, sid, "lore", "Central conflict", description=bible.central_conflict, importance=8)
        S.upsert_edge(W, sid, conflict, self.story_node_id, "appears_in")
        for fid in fac_ids.values():
            S.upsert_edge(W, sid, conflict, fid, "concerns")
        for npc in bible.npcs:
            desc = ", ".join(x for x in (npc.role, npc.description) if x)
            state = {"status": "alive", "attitude": npc.attitude}
            if npc.goal:
                state["goal"] = npc.goal
            nid = self._get_or_create("character", npc.name, desc, state, importance=6)
            if npc.faction and npc.faction.lower() in fac_ids:
                S.upsert_edge(W, sid, nid, fac_ids[npc.faction.lower()], "member_of")
            if npc.location and npc.location.lower() in loc_ids:
                S.replace_edge(W, sid, nid, "located_at", loc_ids[npc.location.lower()], 0)
            sentiment = {"friendly": 3, "hostile": -4}.get(npc.attitude.lower().strip(), 0)
            if sentiment:
                S.upsert_edge(W, sid, nid, self.player_id, "regards", sentiment=sentiment, note=npc.attitude)
        start = loc_ids.get(bible.starting_location.lower())
        if start:
            S.replace_edge(W, sid, self.player_id, "located_at", start, 0)

    def load_story(self, story_id: int):
        self.wait_ready()
        st = self.store.get_story(story_id)
        if not st:
            raise KeyError(story_id)
        self.story_id, self.world_id = story_id, st["world_id"]
        self.character = Character.model_validate(st["player"])
        self.arc = ArcPlan.model_validate(st["arc"]) if st["arc"] else None
        meta = st["meta"]
        self.bible = meta.get("bible", {})
        self.story_node_id, self.player_id = meta.get("story_node_id"), meta.get("player_id")
        self.encounter = Encounter.model_validate(meta["encounter"]) if meta.get("encounter") else None
        turns = self.store.turns(story_id, last=1)
        self.turn = turns[-1]["turn"] if turns else 0
        self._last_plan_turn = self.turn
        self.last_outcome = None
        if turns:
            t = turns[-1]
            self.last_outcome = TurnOutcome(
                t["turn"], t["action"], t["narration"], [Choice.model_validate(c) for c in t["choices"]],
                CheckResult.model_validate(t["check_json"]) if t["check_json"] else None, t["notes"], t["segment_id"])
        self.memory_status, self.memory_error = "idle", None
        self.embed_stale()

    # ================================================================== context helpers
    def open_threads(self) -> list[Node]:
        return [n for n in self.store.nodes(self.world_id, types=("thread",), story_id=self.story_id)
                if n.status in OPEN_THREAD]

    def pending_consequences(self, limit: int = 10) -> list[Node]:
        """Active clocks across the whole world, so choices in one tale can still come due in the next."""
        return self.clocks.active(self.world_id)[:limit]

    def dormant_entities(self, limit: int = 6) -> list[Node]:
        out = []
        for n in self.store.nodes(self.world_id, types=("character", "creature", "faction")):
            if n.status.lower() in GONE_STATUSES or n.importance < 6:
                continue
            here = n.last_seen_story == self.story_id
            if (here and self.turn - n.last_seen_turn >= 4) or (not here and n.importance >= 7):
                out.append(n)
        out.sort(key=lambda n: -n.importance)
        return out[:limit]

    def gone_entities(self) -> list[Node]:
        gone = [n for n in self.store.nodes(self.world_id, types=ENTITY_NODE_TYPES)
                if n.status.lower() in GONE_STATUSES and n.type != "player"]
        return sorted(gone, key=lambda n: -n.importance)

    def precheck(self, action: str) -> list[str]:
        """Rules-level precondition check of the player's action, BEFORE narration.

        Adversarial probes showed the narrator follows a false premise in the action ("I ask <dead NPC>",
        "I raise my <item I don't have>") whatever the ledger says, and a post-hoc sentence fix cannot save a
        scene built on that premise. So code detects the premise and hands the narrator a binding correction.
        """
        out: list[str] = []
        if not settings.action_precheck:
            return out
        named = self.retriever.named(self.world_id, action)
        for n in self.store.get_nodes(named).values():
            if n.type in ("character", "creature") and n.status.lower() in GONE_STATUSES:
                out.append(f"{n.name} is {n.status} (since turn {n.updated_turn}). They cannot answer, act or "
                           f"appear alive. Narrate the hero finding only their body, grave or absence instead.")
        for m in _POSSESSIVE.finditer(action):
            phrase = m.group(1).strip(" ,.")
            head = phrase.split()[-1].lower()
            if head in _NOT_ITEMS or phrase.lower() in _NOT_ITEMS:
                continue
            if self.character.find_item(phrase) or self.character.find_item(head):
                continue
            out.append(f"The hero does NOT carry a '{phrase}'. They reach for it and find nothing; narrate "
                       f"that, never its use.")
        return out

    def _present_names(self) -> list[str]:
        """Living characters at the hero's location or mentioned in the last scene."""
        S, W = self.store, self.world_id
        loc = self.resolver.resolve(W, self.character.location, "location") if self.character.location else None
        ids = {e.src for e in S.edges_of(loc.id) if e.rel == "located_at" and e.dst == loc.id} if loc else set()
        last = S.turns(self.story_id, last=1)
        if last and last[-1].get("segment_id"):
            ids |= {e.dst for e in S.edges_of(last[-1]["segment_id"]) if e.rel == "mentions"}
        nodes = [n for n in S.get_nodes(ids).values() if n.type in ("character", "creature")
                 and n.status.lower() not in GONE_STATUSES]
        return [n.name for n in sorted(nodes, key=lambda n: -n.importance)][:10]

    def canon(self) -> str:
        """The LEDGER (PLAN.md §8 B1): authoritative current state, placed last in the narrator prompt."""
        gone = self.gone_entities()
        dead_line = ("Dead or gone (cannot act or speak; may appear only as corpses, memories or ghosts): " +
                     "; ".join(f"{n.name} is {n.status}" for n in gone[:15])) if gone else ""
        c = self.character
        if not settings.use_ledger:                          # phase-1 behaviour, kept for ablation
            return "\n".join(x for x in [dead_line, f"The hero is currently at: {c.location or 'unknown'}."] if x)
        S, W = self.store, self.world_id
        ents = {n.id: n for n in S.nodes(W, types=("character", "creature")) if n.status.lower() not in GONE_STATUSES}
        loc = self.resolver.resolve(W, c.location, "location") if c.location else None
        located: dict[int, int] = {}
        for e in S.edges(W):
            if e.rel == "located_at":
                located[e.src] = e.dst
        here = {nid for nid, lid in located.items() if loc and lid == loc.id}
        last = S.turns(self.story_id, last=1)
        if last and last[-1].get("segment_id"):
            here |= {e.dst for e in S.edges_of(last[-1]["segment_id"]) if e.rel == "mentions"}
        present = [ents[i] for i in here if i in ents]
        present.sort(key=lambda n: -n.importance)
        names = {n.id: n.name for n in S.get_nodes(set(located.values())).values()}

        def label(n: Node) -> str:
            bits = [n.status or "alive"] + ([n.state["attitude"]] if n.state.get("attitude") else [])
            return f"{n.name} ({', '.join(bits)})"

        lines = [f"Hero: {c.name}, {c.hp}/{c.max_hp} HP, at {c.location or 'unknown'}. "
                 f"Carries ONLY: {', '.join(c.inventory) or 'nothing'}. Gold: {c.gold}."]
        if present:
            lines.append("Present or just seen: " + "; ".join(label(n) for n in present[:8]))
        elsewhere = [n for n in ents.values() if n.id not in here and n.id in located and n.importance >= 6
                     and (not loc or located[n.id] != loc.id)]
        elsewhere.sort(key=lambda n: -n.importance)
        if elsewhere:
            lines.append("Elsewhere (last known; they must arrive before appearing): " +
                         "; ".join(f"{n.name} at {names.get(located[n.id], '?')}" for n in elsewhere[:6]))
        if dead_line:
            lines.append(dead_line)
        if self.encounter and not self.encounter.over:
            lines.append("In combat: " + self.encounter.status_text())
        near = [k for k in self.clocks.active(W)
                if int(k.state.get("filled", 0)) * 2 >= int(k.state.get("segments", 4))]
        if near:
            lines.append("Threats closing in: " + "; ".join(describe(k) for k in near[:3]))
        return "\n".join(lines)

    def summaries(self) -> list[Node]:
        return [n for n in self.store.nodes(self.world_id, types=("summary",), story_id=self.story_id)]

    def story_so_far(self) -> str:
        sums = self.summaries()
        arcs = [s for s in sums if s.state.get("level") == 2]
        chapters = [s for s in sums if s.state.get("level") == 1 and not s.state.get("rolled")]
        parts = [f"(Earlier) {s.content}" for s in arcs] + \
                [f"(Turns {s.state.get('start')}-{s.state.get('end')}) {s.content}" for s in chapters]
        covered = max([s.state.get("end", -1) for s in sums], default=-1)
        window_start = self.turn - settings.recent_scenes + 1
        gap_events = [e for e in self.store.nodes(self.world_id, types=("event",), story_id=self.story_id)
                      if covered < e.turn < window_start]
        if gap_events:
            parts.append("Since then: " + " ".join(e.description for e in gap_events[-12:]))
        text = "\n".join(parts)
        cap = settings.summary_token_budget * 4
        return text[-cap:] if len(text) > cap else text

    def recent_scenes(self) -> list[dict]:
        return self.store.turns(self.story_id, last=settings.recent_scenes)

    # ================================================================== turn pipeline
    def prepare_opening(self) -> TurnPrep:
        ctx = SceneContext(world=bible_text(self.bible), sheet=self.character.sheet_text(), story_so_far="",
                           memory=self._recall(f"{self.bible.get('opening_hook', '')} {self.character.location}",
                                               0, []).context,
                           canon=self.canon(), director=Director.notes(self.arc), opening=True,
                           location=self.character.location, hook=self.bible.get("opening_hook", ""))
        return TurnPrep(turn=0, action=OPENING_ACTION, ctx=ctx, recall=self.last_recall)

    def _recall(self, query: str, turn: int, exclude: list[int]) -> RecallResult:
        focus = [self.player_id]
        loc = self.resolver.resolve(self.world_id, self.character.location, "location") \
            if self.character.location else None
        if loc:
            focus.append(loc.id)
        titles = {s["id"]: s["title"] for s in self.store.list_stories(self.world_id)}
        self.last_recall = self.retriever.recall(self.world_id, self.story_id, turn, query, focus_ids=focus,
                                                 exclude_ids=exclude + [self.story_node_id],
                                                 story_titles=titles, player_id=self.player_id)
        return self.last_recall

    def prepare_turn(self, action: str, choice: Choice | None = None) -> TurnPrep:
        self.wait_ready()
        turn = self.turn + 1
        recent = self.recent_scenes()
        if self.encounter and not self.encounter.over:
            return self._prepare_combat_turn(action, choice, turn, recent)
        last_prose = recent[-1]["narration"] if recent else ""
        notes: list[str] = []
        if wants_combat(action):              # the hero starts a fight: the encounter engine takes over
            present = self._present_names()
            seeds = self.adjudicator.opponents(action, last_prose, ", ".join(present))
            if seeds:
                start_notes = self._start_encounter(seeds, turn)
                if self.encounter:
                    prep = self._prepare_combat_turn(action, None, turn, recent)
                    prep.notes = start_notes + prep.notes
                    return prep

        check: Optional[CheckResult] = None
        if choice is not None and choice.ability and choice.dc:
            check = roll_check(self.character, choice.ability, choice.dc, danger=choice.ability in ("STR", "DEX", "CON"),
                               reason="chosen option", rng=self.rng)
        elif choice is None:
            adj = self.adjudicator.judge(action, last_prose, self.character.sheet_text())
            if adj.needs_check:
                check = roll_check(self.character, adj.ability, adj.dc, adj.danger, adj.reason, rng=self.rng)
        if check and check.damage:
            notes += self.character.apply(hp_change=-check.damage)

        arc = self.arc or ArcPlan()
        query = "\n".join(x for x in [action, last_prose[-600:], f"Location: {self.character.location}",
                                      check.reason if check else "", " ".join(arc.callbacks),
                                      arc.next_beats[0] if arc.next_beats else ""] if x)
        exclude = [t["segment_id"] for t in recent if t.get("segment_id")]
        recall = self._recall(query, turn, exclude)

        if check and check.outcome in ("failure", "critical failure"):   # failures feed the threats
            scene_ids = set(recall.included) | set(self.retriever.named(self.world_id, action))
            clock = self.clocks.most_relevant(self.world_id, scene_ids)
            # an ordinary failure feeds a specific threat; only a critical failure feeds the campaign front
            if clock and (not clock.state.get("front") or check.outcome == "critical failure"):
                _, tick_notes = self.clocks.tick(clock.id, 2 if check.outcome == "critical failure" else 1,
                                                 self.story_id, turn, f"failed {check.ability} check",
                                                 source="failure")
                notes += tick_notes
        director = Director.notes(self.arc)
        due = self.clocks.due(self.world_id)[:1]
        if due:
            c = due[0]
            director = (f"MANDATORY THIS SCENE: {c.state.get('payoff', c.name)} — this is the consequence "
                        f"'{c.name}' coming due. Make it happen on screen now, plausibly, building on the "
                        "current situation.\n" + director)

        ctx = SceneContext(world=bible_text(self.bible), sheet=self.character.sheet_text(),
                           story_so_far=self.story_so_far(), memory=recall.context, canon=self.canon(),
                           director=director,
                           recent=[(t["action"], t["narration"]) for t in recent], action=action,
                           dice=check.describe() if check else "",
                           hero_down=self.character.is_down or self.downed_pending,
                           rules_check=self.precheck(action))
        return TurnPrep(turn=turn, action=action, ctx=ctx, recall=recall, check=check, notes=notes,
                        firing=[c.id for c in due])

    # ================================================================== combat phase (PLAN.md §8D)
    def _save_encounter(self):
        meta = self.store.get_story(self.story_id)["meta"]
        meta["encounter"] = self.encounter.model_dump() if self.encounter else None
        self.store.update_story(self.story_id, meta=meta)

    def _prepare_combat_turn(self, action: str, choice: Choice | None, turn: int, recent: list[dict]) -> TurnPrep:
        enc = self.encounter
        key = choice.combat if choice is not None and choice.combat else classify(action, enc)
        rnd = enc.round
        lines = resolve_round(enc, self.character, key, self.rng)      # code decides every hit and death
        status = enc.status_text()
        notes = list(lines)
        outcome = enc.outcome if enc.over else ""
        if enc.over:
            notes += self._end_encounter(turn)
        else:
            self._save_encounter()
        query = action + "\n" + " ".join(e.name for e in (enc.enemies if enc else []))
        exclude = [t["segment_id"] for t in recent if t.get("segment_id")]
        recall = self._recall(query, turn, exclude)
        ctx = SceneContext(world=bible_text(self.bible), sheet=self.character.sheet_text(),
                           story_so_far=self.story_so_far(), memory=recall.context, canon=self.canon(),
                           director="" if not outcome else Director.notes(self.arc),
                           recent=[(t["action"], t["narration"]) for t in recent], action=action,
                           combat_round=rnd, combat_results="\n".join(lines), combat_status=status,
                           combat_outcome=outcome, hero_down=self.character.is_down,
                           rules_check=self.precheck(action))
        return TurnPrep(turn=turn, action=action, ctx=ctx, recall=recall, notes=notes, combat=True)

    def _start_encounter(self, seeds, turn: int, segment_id: int | None = None) -> list[str]:
        """Create an encounter. Called before narration when the hero attacks, or after a scene in which
        enemies attacked (then `segment_id` is that scene and its choices are replaced by the combat menu)."""
        S, W, sid = self.store, self.world_id, self.story_id
        if turn - self._last_downed_turn < 3:      # the hero was just beaten: no fresh fight while recovering
            return []
        allow_boss = (self.arc.current_act == 3 if self.arc else False) or self.character.level >= 4
        seeds = budget_seeds(seeds, self.character.level, allow_boss)
        loc = self.resolver.resolve(W, self.character.location, "location") if self.character.location else None
        enemies, used = [], set()
        for sd in seeds[:6]:
            node = self.resolver.resolve(W, sd.name)
            if node is not None and node.type in ("character", "creature") and node.id != self.player_id \
                    and node.id not in used:
                if node.status.lower() in GONE_STATUSES:
                    continue
                nid = node.id
            else:
                nid = S.add_node(W, sid, "character", sd.name, description=f"hostile {sd.tier} combatant",
                                 state={"status": "alive", "attitude": "hostile"}, turn=turn,
                                 importance={"minion": 3, "standard": 4, "elite": 6, "boss": 8}[sd.tier])
                S.upsert_edge(W, sid, nid, self.story_node_id, "appears_in", turn=turn)
                if segment_id:
                    S.upsert_edge(W, sid, segment_id, nid, "mentions", turn=turn)
                if loc:
                    S.replace_edge(W, sid, nid, "located_at", loc.id, turn)
            used.add(nid)
            S.upsert_edge(W, sid, nid, self.player_id, "fights", sentiment=-6, turn=turn)
            enemies.append(make_enemy(sd.name, sd.tier, self.character.level, nid))
        if not enemies:
            return []
        self.encounter = Encounter(enemies=enemies, start_turn=turn, location=self.character.location)
        self._save_encounter()
        if segment_id:
            choices = combat_choices(self.encounter, self.character)
            S.update_turn_choices(sid, turn, [c.model_dump() for c in choices])
            if self.last_outcome and self.last_outcome.turn == turn:
                self.last_outcome.choices = choices
        return ["⚔ Combat begins: " + ", ".join(f"{e.name} ({e.tier})" for e in enemies)]

    def _end_encounter(self, turn: int) -> list[str]:
        """Archive the encounter and write its outcome to memory deterministically."""
        enc, S, W, sid = self.encounter, self.store, self.world_id, self.story_id
        status_map = {"defeated": "dead", "fled": "fled", "surrendered": "captured"}
        for e in enc.enemies:
            if e.node_id and e.status in status_map:
                S.update_node(e.node_id, sid, turn, state={"status": status_map[e.status]})
        parts = "; ".join(f"{e.name} {e.status if e.status != 'fighting' else 'still standing'}" for e in enc.enemies)
        result = {"victory": "won", "parley": "forced a surrender", "fled": "escaped", "downed": "was beaten"}[enc.outcome]
        summary = f"{self.character.name} fought at {enc.location or 'an unknown place'} and {result} ({parts})."
        eid = S.add_node(W, sid, "event", summary[:90], description=summary, turn=turn,
                         importance=8 if any(e.tier == "boss" for e in enc.enemies) else 6)
        S.upsert_edge(W, sid, eid, self.player_id, "involves", turn=turn)
        for e in enc.enemies:
            if e.node_id:
                S.upsert_edge(W, sid, eid, e.node_id, "involves", turn=turn)
        loc = self.resolver.resolve(W, enc.location, "location") if enc.location else None
        if loc:
            S.upsert_edge(W, sid, eid, loc.id, "occurred_at", turn=turn)
        notes = [f"⚔ Fight over: {enc.outcome}"]
        if enc.outcome == "downed":
            self._last_downed_turn = turn
        if enc.outcome in ("victory", "parley"):
            notes += self.character.apply(xp_gain=enc.xp())
        meta = S.get_story(sid)["meta"]
        meta.setdefault("past_encounters", []).append(
            {"start": enc.start_turn, "end": turn, "outcome": enc.outcome, "summary": summary})
        meta["encounter"] = None
        S.update_story(sid, meta=meta)
        self.encounter = None
        return notes

    def narrate(self, prep: TurnPrep) -> Iterator[str]:
        """Stream the scene; when the stream is exhausted the turn is saved and memory commit starts."""
        filt = StreamFilter([p for _, p in prep.ctx.recent])
        for piece in self.narrator.stream(prep.ctx):
            shown = filt.feed(piece)
            if shown:
                yield shown
        tail = filt.finish()
        if tail:
            yield tail
        self._finalize(prep, filt.raw, filt.text)

    def play(self, action: str, choice: Choice | None = None) -> TurnOutcome:
        """Non-streaming convenience (CLI / tests)."""
        prep = self.prepare_turn(action, choice)
        for _ in self.narrate(prep):
            pass
        return self.last_outcome

    def start(self) -> TurnOutcome:
        prep = self.prepare_opening()
        for _ in self.narrate(prep):
            pass
        return self.last_outcome

    def _finalize(self, prep: TurnPrep, raw: str, display: str = ""):
        parsed_prose, choices = split_narration(raw)
        prose = scrub(display or parsed_prose, [p for _, p in prep.ctx.recent], prep.action)             or raw.strip() or "The world holds its breath."
        if self.encounter and not self.encounter.over:          # mid-fight: the menu comes from the rules
            choices = combat_choices(self.encounter, self.character)
        elif len(choices) < 2:
            choices = FALLBACK_CHOICES
        S, W, sid, T = self.store, self.world_id, self.story_id, prep.turn
        seg = S.add_node(W, sid, "segment", f"Scene {T}", content=prose,
                         description=prose[:200], importance=3, turn=T)
        prev = S.turns(sid, last=1)
        if prev and prev[-1].get("segment_id"):
            S.upsert_edge(W, sid, prev[-1]["segment_id"], seg, "next", turn=T)
        if self.character.is_down or prep.ctx.hero_down:   # revived with a setback, as narrated
            self.character.hp = max(1, self.character.hp)
            self.downed_pending = False
            prep.notes.append("You come to with 1 HP")
        self.turn = T
        S.save_turn(sid, T, action=prep.action, check=prep.check.model_dump() if prep.check else None,
                    narration=prose, choices=[c.model_dump() for c in choices],
                    retrieval=prep.recall.to_log() if prep.recall else None, notes=prep.notes, segment_id=seg)
        self._save_player()
        self.last_outcome = TurnOutcome(T, prep.action, prose, choices, prep.check, list(prep.notes), seg)
        rec = SceneRecord(world_id=W, story_id=sid, story_node_id=self.story_node_id, turn=T, action=prep.action,
                          dice=f"DICE RESULT: {prep.check.describe()}" if prep.check else "",
                          damage=prep.check.damage if prep.check else 0, narration=prose, segment_id=seg,
                          player_id=self.player_id, hero_name=self.character.name, combat=prep.combat)
        known = [r.node for r in (prep.recall.items if prep.recall else []) if r.node.is_entity][:30]
        self.memory_status, self.memory_error = "recording", None
        self._bg = threading.Thread(target=self._commit, args=(rec, known, prep.notes, prep.firing, prep.ctx.canon,
                                                                bool(prep.ctx.rules_check)),
                                    daemon=True)
        self._bg.start()

    # ================================================================== background memory commit
    def _risky(self, narration: str, gone: list[Node], flagged: bool) -> bool:
        """Risk signals that justify an LLM continuity check (PLAN.md §12): a dead character is named, the
        precheck corrected the action, or the hero uses / draws / drinks something."""
        if flagged or _USE_VERBS.search(narration):
            return True
        # Merely naming a dead character is common (graves, memories: 11/15 scenes in one story); the
        # deterministic detector (dead name as the subject of a verb, or speaking) is the useful signal.
        return bool(self.continuity.rule_hints(narration, gone))

    def _check_continuity(self, rec: SceneRecord, ledger: str, notes: list[str], flagged: bool = False):
        """Check the scene against the ledger and repair contradicting sentences before memory sees them."""
        gone = [n for n in self.gone_entities() if n.type in ("character", "creature")]
        if settings.gate_checker and not self._risky(rec.narration, gone, flagged):
            self.continuity_log.append({"turn": rec.turn, "hints": [], "applied": [], "violations": [],
                                        "skipped": True})
            return
        self.memory_status = "checking continuity"
        hints = self.continuity.rule_hints(rec.narration, gone)
        violations = self.continuity.check(rec.narration, ledger, hints, gone=gone,
                                           inventory=list(self.character.inventory))
        fixed, applied = self.continuity.apply(rec.narration, violations) if violations else (rec.narration, [])
        remaining = self.continuity.rule_hints(fixed, gone)
        if len(remaining) >= 2:                    # the scene is built on the contradiction: rewrite it once
            rewritten = self.continuity.rewrite(fixed, ledger, remaining, gone)
            if rewritten:
                fixed = rewritten
                applied.append("scene rewritten: " + "; ".join(sorted({h.split(" is ")[0] for h in remaining})))
        if applied and fixed != rec.narration:
            rec.narration = fixed
            self.store.update_turn_narration(rec.story_id, rec.turn, fixed)
            self.store.update_node(rec.segment_id, rec.story_id, rec.turn, content=fixed, description=fixed[:200])
            if self.last_outcome and self.last_outcome.turn == rec.turn:
                self.last_outcome.prose = fixed
            notes += [f"🛠 Continuity fix: {a}" for a in applied]
        self.continuity_log.append({"turn": rec.turn, "hints": hints, "applied": applied,
                                    "violations": [v.model_dump() for v in violations]})

    def _commit(self, rec: SceneRecord, known: list[Node], notes: list[str], firing: list[int] = (),
                ledger: str = "", flagged: bool = False):
        t0 = time.time()
        try:
            if settings.continuity_check and rec.turn > 0:
                self._check_continuity(rec, ledger, notes, flagged)
                self.memory_status = "recording"
            for t in self.store.turns(rec.story_id, last=settings.recent_scenes):   # entities in recent scenes
                if t.get("segment_id"):
                    ids = [e.dst for e in self.store.edges_of(t["segment_id"]) if e.rel == "mentions"]
                    known += [n for n in self.store.get_nodes(ids).values() if n not in known]
            threads, clocks = self.open_threads(), self.pending_consequences()
            ex = self.chronicler.extract(rec, known, threads, clocks)
            protected = {e.node_id for e in self.encounter.enemies if e.node_id} if self.encounter else set()
            res = self.chronicler.commit(rec, ex, threads, clocks, inventory=list(self.character.inventory),
                                         protected=protected)
            if rec.combat:                         # the combat engine owns HP while fighting
                res.player.hp_change = 0
            elif rec.turn == 0:                    # the opening never costs the hero HP or gear
                res.player.hp_change = max(0, res.player.hp_change)
                res.player.items_lost = []
            else:                                  # real harm comes from dice and combat; prose adds at most a scratch
                res.player.hp_change = max(-3, res.player.hp_change)
            if not rec.combat and ex.combat_starts and not self.encounter:
                res.notes += self._start_encounter(ex.combat_starts, rec.turn, rec.segment_id)
            for cid in firing:                     # the mandated payoff was narrated: the clock has fired
                res.notes += self.clocks.fire(cid, rec.world_id, rec.story_id, rec.turn, res.event_ids)
                res.consequences_resolved += 1
            p = res.player
            notes += self.character.apply(hp_change=p.hp_change, gold_change=p.gold_change, xp_gain=p.xp_gain,
                                          items_gained=p.items_gained, items_lost=p.items_lost, location=p.location)
            if self.character.is_down:             # narrate the collapse in the next scene
                self.character.hp = 1
                self.downed_pending = True
            self._save_player()
            notes += res.notes
            self.store.update_turn_notes(rec.story_id, rec.turn, notes)
            self.last_commit = res
            self.embed_stale()

            if rec.turn > 0 and rec.turn % settings.summarize_every == 0:
                self.memory_status = "summarizing"
                self._summarize(rec.turn)
            since = rec.turn - self._last_plan_turn
            if rec.turn > 0 and (since >= settings.replan_every or
                                 ((res.thread_closed or res.consequences_resolved) and since >= 2)):
                self.memory_status = "replanning"
                plan_notes = self.replan()
                if plan_notes:
                    self.store.update_turn_notes(rec.story_id, rec.turn, notes + plan_notes)
            self.memory_status = "idle"
            log.info("commit turn %d in %.1fs", rec.turn, time.time() - t0)
            self.warm_next()
        except Exception as e:  # noqa: BLE001 - background thread must never kill the game
            log.exception("memory commit failed")
            self.memory_status, self.memory_error = "error", str(e)

    def warm_next(self):
        """Prefill the KV cache with the next narrator prompt's stable prefix (PLAN.md §12)."""
        if not settings.warm_prefix or self.story_id is None:
            return
        try:
            recent = [(t["action"], t["narration"]) for t in self.recent_scenes()]
            self.narrator.warm(bible_text(self.bible), self.story_so_far(), recent)
        except Exception as e:  # noqa: BLE001 - an optimisation must never break a turn
            log.warning("prefix warm-up failed: %s", e)

    def _summarize(self, turn: int):
        sums = self.summaries()
        start = max([s.state.get("end", -1) for s in sums], default=-1) + 1
        scenes = [(t["turn"], t["action"], t["narration"]) for t in self.store.turns(self.story_id)
                  if start <= t["turn"] <= turn]
        if not scenes:
            return
        text = self.summarizer.chapter(scenes, self.character.name)
        n_ch = len([s for s in sums if s.state.get("level") == 1]) + 1
        nid = self.store.add_node(self.world_id, self.story_id, "summary", f"Chapter {n_ch}", content=text,
                                  description=text[:200], importance=6, turn=turn,
                                  state={"level": 1, "start": start, "end": turn, "rolled": False})
        mentioned = set()
        for t in self.store.turns(self.story_id):
            if start <= t["turn"] <= turn and t.get("segment_id"):
                mentioned |= {e.dst for e in self.store.edges_of(t["segment_id"]) if e.rel == "mentions"}
        for m in mentioned:
            self.store.upsert_edge(self.world_id, self.story_id, nid, m, "mentions", turn=turn)
        chapters = [s for s in self.summaries() if s.state.get("level") == 1 and not s.state.get("rolled")]
        if len(chapters) > settings.arc_rollup_after:
            roll = chapters[:-2]
            arc_text = self.summarizer.arc([c.content for c in roll])
            self.store.add_node(self.world_id, self.story_id, "summary", f"Arc to turn {roll[-1].state['end']}",
                                content=arc_text, description=arc_text[:200], importance=7, turn=turn,
                                state={"level": 2, "start": roll[0].state["start"], "end": roll[-1].state["end"]})
            for c in roll:
                self.store.update_node(c.id, self.story_id, turn, state={"rolled": True})
        self.embed_stale()

    def replan(self) -> list[str]:
        notes: list[str] = []
        for front in self.clocks.fronts(self.world_id):         # time passes: the threat advances
            if front.state.get("status") == "ticking":
                notes += self.clocks.tick(front.id, 1, self.story_id, self.turn, "time passes", source="time")[1]
        recent = "\n".join(f"Hero: {t['action']} → {t['narration'][:300]}" for t in self.recent_scenes())
        self.arc = self.director.plan(world=bible_text(self.bible, include_secrets=True),
                                      sheet=self.character.sheet_text(), turn=self.turn,
                                      story_so_far=self.story_so_far(), recent=recent, threads=self.open_threads(),
                                      pending=self.pending_consequences(), dormant=self.dormant_entities(),
                                      previous=self.arc)
        self._last_plan_turn = self.turn
        self.store.update_story(self.story_id, arc=self.arc.model_dump())
        return notes

    # ================================================================== export
    def export_markdown(self) -> str:
        st = self.store.get_story(self.story_id)
        out = [f"# {st['title']}", f"*{self.bible.get('world_name', '')} — theme: {st['theme']}*", ""]
        for t in self.store.turns(self.story_id):
            if t["turn"] > 0:
                out.append(f"> **{self.character.name}:** {t['action']}")
                if t["check_json"]:
                    c = CheckResult.model_validate(t["check_json"])
                    out.append(f"> *{c.badge()}*")
                out.append("")
            out += [t["narration"], ""]
        return "\n".join(out)
