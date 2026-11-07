"""End-to-end engine test with a scripted fake LLM: no Ollama needed."""
from conftest import fake_embed

from loreweaver.engine import GameEngine
from loreweaver.llm import CallStats
from loreweaver.memory import MemoryStore
from loreweaver.schemas import Adjudication, ArcPlan, Extraction, WorldBible

BIBLE = {
    "title": "The Pale Lantern", "world_name": "Halcyon Reach", "tone": "salt-stained noir",
    "setting": "A storm-wracked archipelago of smugglers.", "central_conflict": "A cult seized the lighthouse.",
    "locations": [{"name": "Port Halcyon", "description": "harbor town"},
                  {"name": "The Lighthouse", "description": "cult stronghold"}],
    "factions": [{"name": "Pale Lantern Cult", "description": "drowned-god cult", "goal": "wake the god"}],
    "npcs": [{"name": "Silas Crane", "role": "harbormaster", "description": "nervous, in debt", "goal": "survive",
              "secret": "owes the cult", "attitude": "neutral", "faction": "Pale Lantern Cult",
              "location": "Port Halcyon"},
             {"name": "Oren", "role": "first mate", "description": "loyal", "attitude": "friendly",
              "location": "Port Halcyon"}],
    "opening_hook": "Your ship limps into port as the lighthouse burns green.", "starting_location": "Port Halcyon",
    "front": {"name": "The Drowned God Wakes", "doom": "The god rises and drowns the port",
              "portents": ["fish die in the harbor", "the bells ring by themselves", "the sea turns black"]},
}

EXTRACTIONS = {
    1: {"entities": [{"name": "Harbormaster Silas", "type": "person", "status": "hostile",
                      "description": "humiliated by the hero"},
                     {"name": "Lighthouse Key", "type": "item", "description": "rusted iron key"}],
        "relations": [{"source": "Silas Crane", "target": "Vex", "relation": "hates", "sentiment": -7}],
        "events": [{"summary": "Vex took the Lighthouse Key from Silas at knifepoint.",
                    "participants": ["Vex", "Silas"], "location": "Port Halcyon", "importance": 7}],
        "threads": [{"title": "Retake the lighthouse", "status": "opened", "related": ["Pale Lantern Cult"]}],
        "decisions": [{"decision": "Vex threatened Silas instead of paying",
                       "consequence": "Silas leads cultists to ambush Vex", "clock": "Silas's revenge",
                       "affected": ["Silas Crane"], "severity": 1}],               # severity 1 -> 4 segments
        "player": {"items_gained": ["Lighthouse Key"], "xp_gain": 20}},
    2: {"entities": [{"name": "Mother Ysolde", "type": "character", "description": "high priestess of the cult"}],
        "relations": [{"source": "Mother Ysolde", "target": "Pale Lantern Cult", "relation": "leads", "sentiment": 8}],
        "events": [{"summary": "Vex glimpsed Mother Ysolde on the lighthouse balcony.", "participants": ["Vex",
                    "Mother Ysolde"], "importance": 6}],
        "threads": [{"title": "Retake the Lighthouse", "status": "advanced"}],
        "clock_ticks": [{"clock": "Silas's revenge", "ticks": 2, "reason": "Silas was seen with cultists"}],
        "player": {"location": "The Lighthouse"}},
    3: {"events": [{"summary": "Oren warned Vex that Silas is talking to the cult.", "participants": ["Oren", "Vex"],
                    "importance": 5}]},
    4: {"clock_ticks": [{"clock": "Silas revenge", "ticks": 2, "reason": "Silas tells the cult your route"}]},
    5: {"entities": [{"name": "Silas", "type": "character", "status": "dead"}],
        "events": [{"summary": "Silas led cultists to ambush Vex and died in the fight.",
                    "participants": ["Silas Crane", "Vex"], "importance": 8}],
        "player": {"hp_change": -3}},
}


NARRATION = {   # scripted scenes that actually mention what EXTRACTIONS claims (grounding checks this)
    0: "Your ship limps into Port Halcyon as the lighthouse burns green. Oren waits on the docks.",
    1: "You press the dagger to Harbormaster Silas's throat. He hands over the Lighthouse Key, hate in his eyes.",
    2: "You creep toward The Lighthouse. On the balcony stands Mother Ysolde, high priestess of the Pale "
       "Lantern Cult. Retake the Lighthouse, you think.",
    3: "Oren grips your arm: Silas has been whispering your route to the cult.",
    4: "At dusk you watch Silas slip into the cult's tent.",
    5: "Silas leads cultists out of the fog. Steel rings; Silas falls and does not rise. You bleed from a cut.",
}


class FakeLLM:
    def __init__(self):
        self.stats = CallStats()
        self.turn = 0
        self.prompts = []

    def warm(self, system, prefix):
        self.warmed = prefix

    def embed(self, texts):
        return fake_embed(texts)

    def stream(self, system, prompt, **kw):
        self.prompts.append(prompt)
        text = (f"{NARRATION.get(self.turn, 'Fog rolls over the docks while Oren keeps watch.')}\n\n"
                "CHOICES:\n1. [DEX 12] Climb the tower\n2. Ask Oren for advice\n3. Wait for nightfall\n")
        for i in range(0, len(text), 16):
            yield text[i:i + 16]

    def complete(self, system, prompt, **kw):
        return "Chapter summary: Vex arrived, robbed Silas, and Silas died."

    def json(self, system, prompt, model_cls, **kw):
        if model_cls is WorldBible:
            return WorldBible.model_validate(BIBLE)
        if model_cls is ArcPlan:
            return ArcPlan(premise="Retake the light", next_beats=["Silas seeks revenge"], callbacks=["Oren"])
        if model_cls is Adjudication:
            return Adjudication(needs_check=True, ability="CHA", dc=10, danger=False, reason="intimidation")
        if model_cls is Extraction:
            return Extraction.model_validate(EXTRACTIONS.get(self.turn, {}))
        return model_cls()                      # e.g. the continuity checker: "no violations"


def test_full_pipeline_builds_connected_memory():
    llm = FakeLLM()
    eng = GameEngine(MemoryStore(":memory:"), llm, seed=7)
    sid = eng.new_story("pirate noir", "Vex", "Rogue")
    out = eng.start()
    assert out.turn == 0 and len(out.choices) == 3 and out.choices[0].ability == "DEX"
    eng.wait_ready()

    for t, action in enumerate(["I threaten Silas with my dagger", "I sneak toward the lighthouse",
                                "I look around", "I wait in the shadows", "I fight the cultists"], start=1):
        llm.turn = t
        o = eng.play(action)
        assert o.turn == t
        assert (o.check is None) == (action in ("I wait in the shadows", "I look around"))   # routine actions skip dice
        assert eng.wait_ready(5) and eng.memory_status == "idle", eng.memory_error

    S, W = eng.store, eng.world_id
    silas = eng.resolver.resolve(W, "Silas Crane", "character")
    assert silas.status == "dead" and "Harbormaster Silas" in silas.aliases          # resolved, not duplicated
    assert len([n for n in S.nodes(W, ("character",)) if "silas" in n.name.lower()]) == 1
    assert [h["new"] for h in S.history(silas.id) if h["field"] == "status"] == ["dead"]
    assert silas.state["attitude"] == "hostile"          # disposition is kept apart from life state
    assert "Lighthouse Key" in eng.character.inventory and eng.character.location == "The Lighthouse"

    threads = S.nodes(W, ("thread",))
    assert len(threads) == 1 and threads[0].status == "advanced"                   # fuzzy thread match
    # decision -> clock -> mandated scene -> fired, linked to what happened (deterministic, never early)
    [decision] = S.nodes(W, ("decision",))
    revenge = next(c for c in S.nodes(W, ("clock",)) if not c.state.get("front"))
    assert revenge.name == "Silas's revenge" and revenge.state["segments"] == 4
    assert revenge.state["status"] == "fired" and decision.status == "resolved"
    assert revenge.state["fired_turn"] > revenge.state["due_turn"]
    assert any(e.rel == "led_to" for e in S.edges_of(decision.id))
    assert any("MANDATORY THIS SCENE: Silas leads cultists to ambush Vex" in p for p in llm.prompts)
    front = next(c for c in S.nodes(W, ("clock",)) if c.state.get("front"))
    assert front.state["filled"] >= 1 and front.state["portents"][0] == "fish die in the harbor"
    assert "Silas Crane is dead" in eng.canon()

    summaries = S.nodes(W, ("summary",))
    assert summaries and summaries[0].state["end"] == 5                           # chapter at turn 5
    assert eng.arc.premise == "Retake the light"

    # memory recall: asking about the key surfaces Silas and the robbery
    res = eng.retriever.recall(W, sid, 6, "What do I do with the Lighthouse Key?", player_id=eng.player_id)
    assert "Silas Crane" in res.context and "knifepoint" in res.context
    # narrator actually received memory + director notes
    assert "## MEMORY" in llm.prompts[-1] and "Work toward this beat now" in llm.prompts[-1]

    # second story in the same world: entities carry over, labelled as a past tale
    llm.turn = 0
    sid2 = eng.new_story("pirate noir sequel", "Mara", "Bard", world_id=W)
    assert sid2 != sid
    oren = eng.resolver.resolve(W, "Oren", "character")
    assert sid2 in S.node_story_ids(oren.id)
    assert S.get_node(silas.id).status == "dead"                                  # not resurrected
    md = eng.export_markdown()
    assert md.startswith("# The Pale Lantern")


class CombatLLM(FakeLLM):
    """Turn 1 ends with two cult guards attacking; later extractions try (and fail) to change HP."""

    def stream(self, system, prompt, **kw):
        self.prompts.append(prompt)
        text = ("Two Cult Guards draw curved blades and charge you across the pier." if self.turn == 1 else
                "Steel flashes on the wet pier as you and the Cult Guards trade blows.")
        text += "\n\nCHOICES:\n1. Run\n2. Fight\n3. Talk"
        yield text

    def json(self, system, prompt, model_cls, **kw):
        if model_cls is Extraction:
            if self.turn == 1:
                return Extraction.model_validate({"combat_starts": [
                    {"name": "Cult Guard 1", "tier": "minion"}, {"name": "Cult Guard 2", "tier": "minion"}]})
            return Extraction.model_validate({"player": {"hp_change": -50}})
        return super().json(system, prompt, model_cls, **kw)


def test_combat_encounter_is_resolved_by_code_and_written_to_memory():
    llm = CombatLLM()
    eng = GameEngine(MemoryStore(":memory:"), llm, seed=3)
    eng.new_story("pirate noir", "Vex", "Fighter")
    eng.start()
    eng.wait_ready()
    llm.turn = 1
    eng.play("I walk down the pier")
    eng.wait_ready()
    assert eng.encounter and len(eng.encounter.enemies) == 2
    assert all(c.combat for c in eng.last_outcome.choices)                  # menu replaced by the rules
    assert eng.store.turns(eng.story_id)[-1]["choices"][0]["combat"].startswith("attack")

    for t in range(2, 30):
        llm.turn = t
        hp_before = eng.character.hp
        attack = next(c for c in eng.last_outcome.choices if (c.combat or "").startswith("attack")) \
            if eng.encounter else eng.last_outcome.choices[0]
        eng.play(attack.text, attack)
        eng.wait_ready()
        assert eng.character.hp > 0 or eng.downed_pending
        assert "COMBAT, round" in llm.prompts[-1] or "THE FIGHT IS OVER" in llm.prompts[-1]
        if eng.encounter is None:
            break
        assert eng.character.hp >= hp_before - 20                             # chronicler's -50 was ignored
    assert eng.encounter is None
    meta = eng.store.get_story(eng.story_id)["meta"]
    assert meta["encounter"] is None and meta["past_encounters"][0]["outcome"] in ("victory", "downed", "fled")
    guards = [n for n in eng.store.nodes(eng.world_id, ("character",)) if n.name.startswith("Cult Guard")]
    assert len(guards) == 2
    if meta["past_encounters"][0]["outcome"] == "victory":
        assert all(g.status in ("dead", "fled") for g in guards)              # written by code, not the LLM
    assert any("fought at" in e.description for e in eng.store.nodes(eng.world_id, ("event",)))


class GhostLLM(FakeLLM):
    """Turn 1 kills Silas; turn 2's narrator wrongly brings him back; the checker catches it."""
    BAD = "Silas Crane strolls out of the tavern and waves at you."

    def stream(self, system, prompt, **kw):
        self.prompts.append(prompt)
        text = {1: "Your blade finds Silas Crane's heart. He falls dead on the pier.",
                2: f"Night falls over Port Halcyon. {self.BAD} Oren curses under his breath."}.get(
            self.turn, "The tide rolls in over Port Halcyon.")
        yield text + "\n\nCHOICES:\n1. Run\n2. Hide\n3. Talk"

    def json(self, system, prompt, model_cls, **kw):
        from loreweaver.agents.continuity import ContinuityReport
        if model_cls is Extraction:
            if self.turn == 1:
                return Extraction.model_validate({"entities": [{"name": "Silas Crane", "type": "character",
                                                                "status": "dead"}]})
            if self.turn == 2:   # the extractor believes the (already repaired) scene shows him alive
                return Extraction.model_validate({"entities": [{"name": "Silas Crane", "type": "character",
                                                                "status": "alive"}]})
        if model_cls is ContinuityReport and self.turn == 2:
            return ContinuityReport.model_validate({"violations": [{
                "kind": "dead_acting", "fact": "Silas Crane is dead", "quote": self.BAD,
                "fix": "A stranger in Silas Crane's old coat strolls out of the tavern and waves at you."}]})
        return super().json(system, prompt, model_cls, **kw)


def test_ledger_checker_and_guards_keep_the_dead_dead():
    llm = GhostLLM()
    eng = GameEngine(MemoryStore(":memory:"), llm, seed=1)
    eng.new_story("pirate noir", "Vex", "Rogue")
    eng.start(); eng.wait_ready()
    llm.turn = 1; eng.play("I look at Silas"); eng.wait_ready()
    llm.turn = 2; eng.play("I wait for nightfall"); eng.wait_ready()
    assert "## LEDGER" in llm.prompts[-1] and "Silas Crane is dead" in llm.prompts[-1]
    assert "Carries ONLY:" in llm.prompts[-1]
    stored = eng.store.turns(eng.story_id)[-1]
    assert GhostLLM.BAD not in stored["narration"] and "A stranger in Silas Crane's old coat" in stored["narration"]
    assert any(n.startswith("🛠 Continuity fix") for n in stored["notes"])
    seg = eng.store.get_node(stored["segment_id"])
    assert "A stranger" in seg.content                                    # memory holds the corrected scene
    silas = eng.resolver.resolve(eng.world_id, "Silas Crane", "character")
    assert silas.status == "dead"                                         # guard refused the resurrection
    assert eng.continuity_log[-1]["applied"] == ["Silas Crane is dead"]


class AggroLLM(FakeLLM):
    def json(self, system, prompt, model_cls, **kw):
        from loreweaver.schemas import Opponents
        if model_cls is Opponents:
            return Opponents.model_validate({"opponents": [{"name": "Bandit Chief", "tier": "elite"}, "Bandit 1"]})
        return super().json(system, prompt, model_cls, **kw)


def test_hero_initiated_attack_starts_an_encounter_before_narration():
    llm = AggroLLM()
    eng = GameEngine(MemoryStore(":memory:"), llm, seed=5)
    eng.new_story("pirate noir", "Vex", "Ranger")
    eng.start(); eng.wait_ready()
    hp_full = eng.character.hp
    llm.turn = 9
    out = eng.play("I draw my bow and shoot the bandit chief")
    eng.wait_ready()
    assert "COMBAT, round 1" in llm.prompts[-1] and "Bandit Chief" in llm.prompts[-1]
    assert eng.encounter is not None and [e.name for e in eng.encounter.enemies] == ["Bandit Chief", "Bandit 1"]
    assert out.notes[0].startswith("⚔ Combat begins")
    assert all(c.combat for c in out.choices)
    assert eng.character.hp <= hp_full                                     # HP moved only by the engine


def test_precheck_corrects_false_premises_before_narration():
    llm = FakeLLM()
    eng = GameEngine(MemoryStore(":memory:"), llm, seed=2)
    eng.new_story("pirate noir", "Vex", "Rogue")
    eng.start(); eng.wait_ready()
    silas = eng.resolver.resolve(eng.world_id, "Silas Crane", "character")
    eng.store.update_node(silas.id, eng.story_id, 1, state={"status": "dead"})
    rules = eng.precheck("I ask Silas for help and raise my Wand of Storms at the cultists")
    assert any("Silas Crane is dead" in r for r in rules)
    assert any("'Wand of Storms'" in r for r in rules)
    assert eng.precheck("I swing my twin daggers with all my strength") == []   # carried items, body words
    llm.turn = 9
    eng.play("I ask Silas for help")
    assert "RULES CHECK" in llm.prompts[-1] and "Silas Crane is dead" in llm.prompts[-1]


def test_warmed_prefix_is_an_exact_prefix_of_the_next_narrator_prompt():
    llm = FakeLLM()
    eng = GameEngine(MemoryStore(":memory:"), llm, seed=4)
    eng.new_story("pirate noir", "Vex", "Rogue")
    eng.start(); eng.wait_ready()
    for t, action in enumerate(["I look around the docks", "I wait for Oren"], start=1):
        llm.turn = t
        warmed = llm.warmed                      # computed at the end of the previous commit
        eng.play(action); eng.wait_ready()
        assert llm.prompts[-1].startswith(warmed) and len(warmed) > 200   # KV cache can be reused


def test_checker_runs_only_on_risk_signals():
    from types import SimpleNamespace
    eng = GameEngine(MemoryStore(":memory:"), FakeLLM(), seed=1)
    silas = SimpleNamespace(name="Silas Crane", aliases=["Silas"], status="dead", type="character")
    assert not eng._risky("Rain lashes the pier. Your blade clatters uselessly. You held your breath.", [silas], False)
    assert eng._risky("You raise your Wand of Storms.", [silas], False)          # the hero uses something
    assert eng._risky("Silas waves from the pier.", [silas], False)              # a dead character acts
    assert not eng._risky("You kneel at Silas's grave.", [silas], False)          # merely named: no check
    assert eng._risky("Rain lashes the pier.", [silas], True)                    # the precheck fired
