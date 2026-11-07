from conftest import embed_all


def _world(store):
    wid = store.create_world("Halcyon", "pirates", {})
    sid = store.create_story(wid, "The Pale Lantern", "pirates", {})
    return wid, sid


def test_node_update_records_temporal_history(store):
    wid, sid = _world(store)
    nid = store.add_node(wid, sid, "character", "Silas", description="Harbormaster", state={"status": "alive"})
    store.update_node(nid, sid, 5, state={"status": "dead"}, aliases_add=["Harbormaster Silas"])
    n = store.get_node(nid)
    assert n.status == "dead" and "Harbormaster Silas" in n.aliases
    h = store.history(nid)
    assert h[0]["old"] == "alive" and h[0]["new"] == "dead" and h[0]["turn"] == 5


def test_edge_upsert_blends_sentiment_and_strengthens(store):
    wid, sid = _world(store)
    a = store.add_node(wid, sid, "character", "Silas")
    b = store.add_node(wid, sid, "player", "Vex")
    store.upsert_edge(wid, sid, a, b, "distrusts", sentiment=-4, turn=1)
    store.upsert_edge(wid, sid, a, b, "distrusts", sentiment=-9, turn=3)
    [e] = store.edges_of(a)
    assert e.sentiment == round(0.4 * -4 + 0.6 * -9) and e.weight == 1.5 and e.updated_turn == 3


def test_fts_keyword_search(store):
    wid, sid = _world(store)
    store.add_node(wid, sid, "item", "Lighthouse Key", description="rusted iron key")
    store.add_node(wid, sid, "character", "Oren", description="first mate")
    hits = store.fts_search(wid, "where is the rusted key?")
    assert hits and store.get_node(hits[0][0]).name == "Lighthouse Key"


def test_resolver_cascade(mem):
    store, _, resolver, _ = mem
    wid, sid = _world(store)
    silas = store.add_node(wid, sid, "character", "Silas Crane", aliases=["the harbormaster"])
    store.add_node(wid, sid, "faction", "Cult of the Pale Lantern")
    assert resolver.resolve(wid, "Harbormaster Silas", "character").id == silas      # token subset
    assert resolver.resolve(wid, "the harbormaster", "character").id == silas        # alias
    assert resolver.resolve(wid, "Silas Crain", "character").id == silas             # fuzzy
    assert resolver.resolve(wid, "Pale Lantern cult", "faction").name == "Cult of the Pale Lantern"
    assert resolver.resolve(wid, "a guard", "character") is None                      # generic never merges
    assert resolver.is_player_ref("you", "Vex") and resolver.is_player_ref("Captain Vex", "Vex")


def test_retriever_finds_connected_memory_via_graph(mem):
    """Asking about the key should surface Silas (its former owner) and his grudge, unmentioned."""
    store, graph, _, retriever = mem
    wid, sid = _world(store)
    hero = store.add_node(wid, sid, "player", "Vex")
    silas = store.add_node(wid, sid, "character", "Silas", description="harbormaster in debt to the cult",
                           importance=7, turn=1)
    key = store.add_node(wid, sid, "item", "Lighthouse Key", description="rusted iron key", turn=2)
    cult = store.add_node(wid, sid, "faction", "Pale Lantern", description="cult that seized the lighthouse")
    ev = store.add_node(wid, sid, "event", "threat", description="Vex threatened Silas and took the Lighthouse Key",
                        importance=8, turn=2)
    for i in range(10):  # noise
        store.add_node(wid, sid, "location", f"Tavern {i}", description="a smoky tavern with sailors drinking ale")
    store.upsert_edge(wid, sid, silas, key, "owned")
    store.upsert_edge(wid, sid, silas, hero, "hates", sentiment=-7)
    store.upsert_edge(wid, sid, silas, cult, "owes_debt_to")
    store.upsert_edge(wid, sid, ev, silas, "involves")
    store.upsert_edge(wid, sid, ev, key, "involves")
    store.upsert_edge(wid, sid, hero, key, "owns")
    embed_all(store, wid)

    res = retriever.recall(wid, sid, 6, "I use the lighthouse key on the iron door", focus_ids=[hero],
                           player_id=hero)
    ranked = [r.node.id for r in res.items]
    assert ranked.index(key) < 3
    assert silas in res.included and ev in res.included
    assert "hates the hero (-7)" in res.context
    assert hero not in ranked
    log = res.to_log()
    assert log["items"][0]["why"]


def test_cross_story_memory_is_labelled(mem):
    store, _, _, retriever = mem
    wid, sid = _world(store)
    villain = store.add_node(wid, sid, "character", "Morgrave", description="lich who ruled the Pale Lantern",
                             importance=9, turn=3)
    sid2 = store.create_story(wid, "Second Tide", "pirates", {})
    embed_all(store, wid)
    res = retriever.recall(wid, sid2, 1, "Morgrave has returned", story_titles={sid: "The Pale Lantern"})
    assert res.items[0].node.id == villain and "other-story" in res.items[0].why
    assert "past tale 'The Pale Lantern'" in res.context


def test_relation_channel_answers_who_questions(mem):
    store, _, _, retriever = mem
    wid, sid = _world(store)
    hero = store.add_node(wid, sid, "player", "Vex")
    silas = store.add_node(wid, sid, "character", "Silas", description="harbormaster")
    oren = store.add_node(wid, sid, "character", "Oren", description="first mate")
    mira = store.add_node(wid, sid, "character", "Mira", description="smuggler")
    for i in range(8):
        store.add_node(wid, sid, "character", f"Sailor {i}", description="a sailor who hates the sea")
    store.upsert_edge(wid, sid, silas, hero, "distrusts", sentiment=-6)
    store.upsert_edge(wid, sid, oren, hero, "is_loyal_to", sentiment=8)
    store.upsert_edge(wid, sid, mira, hero, "resents", sentiment=-7)
    embed_all(store, wid)
    top = lambda q: [r.node.id for r in retriever.recall(wid, sid, 5, q, player_id=hero).items[:2]]  # noqa: E731
    assert top("Who distrusts me?")[0] == silas                     # edge label match
    assert set(top("Who hates me?")) == {silas, mira}               # negative sentiment toward the hero
    assert top("Who is my friend?")[0] == oren                      # positive sentiment


def test_grounding_keeps_hallucinated_facts_out_of_memory(mem):
    from loreweaver.agents.chronicler import Grounder
    from loreweaver.schemas import Extraction
    store, _, resolver, _ = mem
    wid, sid = _world(store)
    store.add_node(wid, sid, "character", "Silas Crane", aliases=["the harbormaster"])
    ex = Extraction.model_validate({
        "entities": [{"name": "Silas Crane", "type": "character"}, {"name": "Lord Vantor", "type": "character"}],
        "relations": [{"source": "the harbormaster", "target": "Vex", "relation": "hates", "sentiment": -5},
                      {"source": "Lord Vantor", "target": "Vex", "relation": "hunts", "sentiment": -8}],
        "events": [{"summary": "Vex robbed Silas.", "participants": ["Vex", "Silas", "Lord Vantor"]}],
        "player": {"items_gained": ["Lighthouse Key", "Crown of Stars"]}})
    text = "You threaten Silas. The harbormaster spits but hands over the rusted lighthouse key."
    dropped = Grounder(resolver, wid, "Vex", text).filter(ex)
    assert [e.name for e in ex.entities] == ["Silas Crane"]
    assert len(ex.relations) == 1 and ex.events[0].participants == ["Vex", "Silas"]
    assert ex.player.items_gained == ["Lighthouse Key"]
    assert any("Lord Vantor" in d for d in dropped) and any("Crown of Stars" in d for d in dropped)


def test_canon_guard_refuses_unexplained_resurrection(mem):
    from loreweaver.agents.chronicler import Chronicler, SceneRecord
    from loreweaver.schemas import Extraction
    store, _, resolver, _ = mem
    wid, sid = _world(store)
    hero = store.add_node(wid, sid, "player", "Vex")
    story_node = store.add_node(wid, sid, "story", "tale")
    silas = store.add_node(wid, sid, "character", "Silas", state={"status": "dead"})
    seg = store.add_node(wid, sid, "segment", "Scene 9", content="...")
    chron = Chronicler(None, store, resolver)

    def commit(narration, inventory=("Dagger",), used=()):
        rec = SceneRecord(wid, sid, story_node, 9, "I look around", "", 0, narration, seg, hero, "Vex")
        ex = Extraction.model_validate({"entities": [{"name": "Silas", "type": "character", "status": "alive"}],
                                        "player": {"items_used": list(used)}})
        return chron.commit(rec, ex, [], [], inventory=list(inventory))

    res = commit("Silas waves at you from the pier, alive and grinning. You gulp a healing potion.",
                 used=["Healing potion"])
    assert store.get_node(silas).status == "dead"
    assert any("Silas is dead" in v for v in res.violations)
    assert any("Healing potion" in v for v in res.violations)              # phantom item
    commit("By some dark rite Silas was resurrected; he waves at you from the pier.")
    assert store.get_node(silas).status == "alive"                         # justified by the text


def test_numbered_individuals_never_merge(mem):
    from loreweaver.memory.resolver import numbers_conflict
    store, _, resolver, _ = mem
    wid, sid = _world(store)
    g1 = store.add_node(wid, sid, "character", "Cult Guard 1")
    assert resolver.resolve(wid, "Cult Guard 2", "character") is None
    assert resolver.resolve(wid, "Cult Guard 1", "character").id == g1
    assert numbers_conflict("Edric II", "Edric III") and not numbers_conflict("the river", "a vivid view")


def test_contradicting_relation_supersedes_the_old_one(mem):
    store, _, _, retriever = mem
    wid, sid = _world(store)
    hero = store.add_node(wid, sid, "player", "Vex")
    mira = store.add_node(wid, sid, "character", "Mira", description="smuggler")
    store.upsert_edge(wid, sid, mira, hero, "ally_of", sentiment=7, turn=2)
    store.upsert_edge(wid, sid, mira, hero, "betrayed", sentiment=-8, turn=9)
    edges = {e.rel: e for e in store.edges_of(mira)}
    assert edges["ally_of"].valid_to == 9 and edges["betrayed"].current          # history kept, state updated
    card = retriever._entity_card(store.get_node(mira), sid, {}, hero)
    assert "betrayed the hero" in card and "ally of" not in card
    store.upsert_edge(wid, sid, mira, hero, "regards", sentiment=5, turn=10)       # reconciliation
    store.upsert_edge(wid, sid, mira, hero, "regards", sentiment=-6, turn=12)      # reversal replaces, not averages
    assert {e.rel: e for e in store.edges_of(mira)}["regards"].sentiment == -6
