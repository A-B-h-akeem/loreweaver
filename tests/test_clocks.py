from loreweaver.clocks import Clocks, bar


def _setup(store):
    wid = store.create_world("W", "t", {})
    sid = store.create_story(wid, "S", "t", {})
    return wid, sid, Clocks(store)


def test_clock_ticks_portents_due_and_fire(store):
    wid, sid, clocks = _setup(store)
    silas = store.add_node(wid, sid, "character", "Silas")
    dec = store.add_node(wid, sid, "decision", "Vex robbed Silas", state={"status": "pending"})
    ev = store.add_node(wid, sid, "event", "ambush", description="Silas ambushed Vex")
    cid = clocks.create(wid, sid, 1, "Silas's revenge", "Silas ambushes Vex", 4, threatens=[silas], decision_id=dec)
    front = clocks.create(wid, sid, 0, "The god wakes", "the port drowns", 8, front=True,
                          portents=["fish die", "bells ring", "black sea"])
    assert clocks.most_relevant(wid, {silas}).id == cid           # threatens someone in the scene
    assert clocks.most_relevant(wid, set()).id == front           # else the front
    assert clocks.tick(front, 2, sid, 1)[0].state["filled"] == 1      # scenes only nudge a front by 1
    node, notes = clocks.tick(front, 1, sid, 2, source="time")
    assert node.state["current_portent"] == "fish die" and any("portent" in n for n in notes)
    clocks.tick(cid, 3, sid, 2)
    assert clocks.tick(cid, 1, sid, 3)[1] == []                   # scene ticks respect a 2-turn cooldown
    node, notes = clocks.tick(cid, 5, sid, 3, source="failure")   # failures bypass it
    assert node.state["status"] == "due" and node.state["filled"] == 4 and bar(node) == "■■■■"
    assert [c.id for c in clocks.due(wid)] == [cid]
    clocks.fire(cid, wid, sid, 4, [ev])
    assert store.get_node(cid).state["status"] == "fired"
    assert store.get_node(dec).state["status"] == "resolved"
    assert {(e.src, e.rel) for e in store.edges_of(ev)} >= {(cid, "led_to"), (dec, "led_to")}
    assert clocks.tick(cid, 1, sid, 5)[1] == []                   # fired clocks no longer tick


def test_front_cannot_fire_before_the_finale(store):
    wid, sid, clocks = _setup(store)
    front = clocks.create(wid, sid, 0, "The god wakes", "the port drowns", 8, front=True)
    node, _ = clocks.tick(front, 20, sid, 5, source="time")
    assert node.state["filled"] == 7 and node.state["status"] == "ticking"     # held one short of doom
    node, _ = clocks.tick(front, 1, sid, 18, source="time")
    assert node.state["status"] == "due"


def test_averted_clock_resolves_decision_without_firing(store):
    wid, sid, clocks = _setup(store)
    dec = store.add_node(wid, sid, "decision", "Vex spared the assassin", state={"status": "pending"})
    cid = clocks.create(wid, sid, 1, "Assassin returns", "the assassin strikes", 6, decision_id=dec)
    clocks.avert(cid, sid, 3, "You convince her to leave the city forever.")
    assert store.get_node(cid).state["status"] == "averted" and clocks.active(wid) == []
    assert store.get_node(dec).state["status"] == "averted"
