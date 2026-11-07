"""Loreweaver — Streamlit front end.  Run:  streamlit run app.py"""
from __future__ import annotations

import logging
import re

import pandas as pd
import streamlit as st

from loreweaver.combat import hero_ac
from loreweaver.config import settings
from loreweaver.engine import GameEngine
from loreweaver.llm import OllamaLLM
from loreweaver.memory import ENTITY_NODE_TYPES, MemoryStore
from loreweaver.memory.retriever import STRUCTURAL_RELS
from loreweaver.rules import ABILITY_NAMES, CLASSES, xp_for_level
from loreweaver.schemas import ABILITIES, Choice
from loreweaver.viz import TYPE_STYLE, build_graph_html

logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
st.set_page_config(page_title="Loreweaver", page_icon=":material/auto_stories:", layout="wide")

THEME_IDEAS = {
    ":material/sailing: Cursed lighthouse": "a haunted lighthouse on a cursed 1920s fishing island",
    ":material/castle: Fallen kingdom": "a kingdom whose king vanished and whose crown now whispers",
    ":material/rocket_launch: Derelict starship": "a derelict generation starship where the crew worships the AI",
    ":material/local_fire_department: Desert heist": "a heist in a floating desert city ruled by a mechanical sultan",
    ":material/forest: Fey wood": "a forest where fey bargains are binding and time runs strangely",
}


# ============================================================================ shared resources
@st.cache_resource
def shared_store() -> MemoryStore:
    return MemoryStore(settings.db_path)


@st.cache_resource
def shared_llm() -> OllamaLLM:
    return OllamaLLM()


@st.cache_data(ttl=30, show_spinner=False)
def health() -> tuple[bool, str]:
    return shared_llm().health()


@st.cache_data(max_entries=6, show_spinner=False)
def graph_html(_store: MemoryStore, world_id: int, version: int, types: tuple, story_id: int | None,
               highlight: tuple, player_id: int | None, dark: bool, structural: bool):
    return build_graph_html(_store, world_id, types=types, story_id=story_id, highlight=highlight,
                            player_id=player_id, dark=dark, show_structural=structural)


ss = st.session_state
if "engine" not in ss:
    ss.engine = GameEngine(shared_store(), shared_llm())
ss.setdefault("view", "home")
ss.setdefault("pending", None)
ss.setdefault("was_busy", False)
eng: GameEngine = ss.engine
store: MemoryStore = eng.store


def md(text: str) -> str:
    """LLM prose -> safe markdown ('$' would otherwise start LaTeX)."""
    return (text or "").replace("$", "\\$")


def queue(action: str, choice: dict | None = None):
    ss.pending = {"kind": "turn", "action": action, "choice": choice}


def dice_badge(check: dict | None):
    if not check:
        return
    ok = check["outcome"] in ("critical success", "strong success", "success", "partial success")
    color = "green" if ok and check["outcome"] != "partial success" else "orange" if ok else "red"
    label = (f"{ABILITY_NAMES[check['ability']]} DC {check['dc']} · rolled {check['roll']}{check['bonus']:+d} = "
             f"{check['total']} · {check['outcome']}")
    if check.get("damage"):
        label += f" · {check['damage']} damage"
    st.badge(label, icon=":material/casino:", color=color)


# ============================================================================ home
def home():
    st.title("Loreweaver")
    st.caption("An AI dungeon master that remembers. Every character, choice and consequence lives in one "
               "connected memory graph, and stories told in the same world remember each other.")
    ok, msg = health()
    if not ok:
        st.error(msg, icon=":material/error:")
        st.caption("Start Ollama, then reload this page.")
        st.stop()

    new_col, load_col = st.columns([3, 2], gap="large")
    with new_col:
        st.subheader("Begin a new tale")
        st.pills("Ideas", list(THEME_IDEAS), key="idea", label_visibility="collapsed",
                 on_change=lambda: ss.update(theme_input=THEME_IDEAS.get(ss.idea) or ss.get("theme_input", "")))
        worlds = store.list_worlds()
        world_opts = {None: "A brand-new world"} | {
            w["id"]: f"{w['name']} ({w['n_stories']} tale{'s' if w['n_stories'] != 1 else ''}, {w['n_nodes']} memories)"
            for w in worlds}
        with st.form("new_story", border=True):
            theme = st.text_area("Theme", key="theme_input", height=90,
                                 placeholder="e.g. a plague city where the dead keep their jobs")
            c1, c2 = st.columns(2)
            name = c1.text_input("Hero name", value="Vex", max_chars=40)
            hero_class = c2.selectbox("Class", list(CLASSES),
                                      format_func=lambda c: f"{c} — {CLASSES[c]['blurb']}", index=1)
            world = st.selectbox("World", list(world_opts), format_func=world_opts.get,
                                 help="Set the tale in an existing world to meet its characters again, "
                                      "with their history and grudges intact.")
            go = st.form_submit_button("Begin adventure", type="primary", icon=":material/swords:")
        if go:
            if not theme.strip() or not name.strip():
                st.warning("Give the tale a theme and your hero a name.", icon=":material/edit:")
            else:
                with st.status("Forging your world…", expanded=True) as status:
                    try:
                        eng.new_story(theme.strip(), name.strip(), hero_class, world_id=world,
                                      progress=lambda m: (status.update(label=m), st.write(m)))
                    except Exception as e:  # noqa: BLE001
                        status.update(label="The world refused to form", state="error")
                        st.exception(e)
                        st.stop()
                    status.update(label="The world is ready", state="complete")
                ss.view, ss.pending = "play", {"kind": "opening"}
                st.rerun()

    with load_col:
        st.subheader("Continue a tale")
        stories = store.list_stories()
        if not stories:
            st.caption("No tales yet.")
        for s in stories[:25]:
            with st.container(border=True):
                st.markdown(f"**{md(s['title'])}**")
                st.caption(f"{md(s['world_name'])} · turn {s['turn']} · {md(s['theme'][:70])}")
                row = st.container(horizontal=True)
                if row.button("Resume", key=f"resume_{s['id']}", icon=":material/play_arrow:", type="primary"):
                    eng.load_story(s["id"])
                    ss.view, ss.pending = "play", None
                    st.rerun()
                with row.popover("Delete", icon=":material/delete:"):
                    st.caption("Scenes, events and decisions of this tale are erased. Characters and places "
                               "stay in the world.")
                    if st.button("Delete permanently", key=f"del_{s['id']}", type="primary"):
                        store.delete_story(s["id"])
                        st.rerun()


# ============================================================================ sidebar
def memory_panel():
    busy = eng.busy
    if busy:
        st.badge(f"Chronicler: {eng.memory_status}…", icon=":material/sync:", color="orange")
    elif eng.memory_status == "error":
        st.badge("Memory error", icon=":material/error:", color="red")
        st.caption(eng.memory_error or "")
    else:
        st.badge("Memory up to date", icon=":material/check_circle:", color="green")
    stats = store.stats(eng.world_id)
    st.caption(f"{sum(stats['nodes'].values())} memories · {stats['edges']} links in this world")
    if ss.was_busy and not busy:          # commit just finished: refresh sheet, codex, graph
        ss.was_busy = False
        st.rerun()
    ss.was_busy = busy


def sidebar():
    c = eng.character
    st_ = store.get_story(eng.story_id)
    with st.sidebar:
        st.subheader(md(st_["title"]))
        st.caption(md(eng.bible.get("world_name", "")))
        st.markdown(f"**{md(c.name)}** · level {c.level} {c.char_class}")
        st.progress(max(0.0, min(1.0, c.hp / c.max_hp)), text=f"HP {c.hp} / {c.max_hp}")
        lo, hi = xp_for_level(c.level), xp_for_level(c.level + 1)
        st.progress((c.xp - lo) / (hi - lo), text=f"XP {c.xp} / {hi}")
        st.markdown(" ".join(f":gray-badge[{a} {c.abilities[a]} ({c.mod(a):+d})]" for a in ABILITIES))
        st.markdown(f":material/paid: {c.gold} gold  \n:material/location_on: {md(c.location or 'unknown')}")
        with st.expander("Inventory", icon=":material/backpack:", expanded=True):
            st.markdown("\n".join(f"- {md(i)}" for i in c.inventory) or "Empty")
        enc = eng.encounter
        if enc and not enc.over:
            with st.container(border=True):
                st.markdown(f":material/swords: **Combat · round {enc.round}** · your AC {hero_ac(c)}")
                for e in enc.enemies:
                    if e.status == "fighting":
                        st.progress(e.hp / e.max_hp, text=f"{md(e.name)} ({e.tier}) {e.hp}/{e.max_hp}")
                    else:
                        st.caption(f"{md(e.name)}: {e.status}")
        clocks = eng.clocks.active(eng.world_id)
        if clocks:
            with st.expander("Threats", icon=":material/hourglass_top:", expanded=True):
                for k in clocks[:6]:
                    seg, fill = int(k.state.get("segments", 4)), int(k.state.get("filled", 0))
                    label = f"{md(k.name)} {fill}/{seg}" + (" · DUE" if k.state.get("status") == "due" else "")
                    st.progress(fill / seg, text=label)
        st.fragment(run_every="2s" if eng.busy else None)(memory_panel)()
        row = st.container(horizontal=True)
        if row.button("Library", icon=":material/library_books:"):
            ss.view = "home"
            st.rerun()
        row.download_button("Export", data=eng.export_markdown(), icon=":material/download:",
                            file_name=re.sub(r"\W+", "_", st_["title"]).strip("_") + ".md", mime="text/markdown")


# ============================================================================ story tab
def run_pending():
    p, ss.pending = ss.pending, None                  # pop first: an interrupted run never replays dice
    try:
        if p["kind"] == "opening":
            with st.chat_message("assistant", avatar=":material/auto_stories:"):
                with st.status(":shimmer[The Dungeon Master sets the scene]", type="compact") as status:
                    prep = eng.prepare_opening()
                    status.update(label="Scene set", state="complete")
                st.write_stream(eng.narrate(prep))
        else:
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(md(p["action"]))
            choice = Choice.model_validate(p["choice"]) if p.get("choice") else None
            with st.chat_message("assistant", avatar=":material/auto_stories:"):
                with st.status(":shimmer[Consulting the dice and the memory graph]", type="compact") as status:
                    if eng.busy:
                        with st.status("Waiting for the chronicler to finish the last scene", type="step"):
                            eng.wait_ready()
                    prep = eng.prepare_turn(p["action"], choice)
                    with st.status("Dice" if prep.check else "No roll needed", type="step", state="complete"):
                        st.write(prep.check.describe() if prep.check else "A routine action.")
                    inc = set(prep.recall.included) if prep.recall else set()
                    recalled = [r.node.name for r in (prep.recall.items if prep.recall else []) if r.node.id in inc]
                    with st.status(f"Recalled {len(recalled)} memories", type="step", state="complete"):
                        st.write(", ".join(md(n[:50]) for n in recalled[:15]) or "Nothing yet.")
                    if prep.ctx.rules_check:
                        with st.status("Rules check corrected your action", type="step", state="complete"):
                            for r in prep.ctx.rules_check:
                                st.write(md(r))
                    status.update(label=f"Recalled {len(recalled)} memories"
                                        + (f" · rolled {prep.check.total}" if prep.check else ""), state="complete")
                if prep.check:
                    dice_badge(prep.check.model_dump())
                st.write_stream(eng.narrate(prep))
    except Exception as e:  # noqa: BLE001
        st.error(f"The Dungeon Master faltered: {e}", icon=":material/error:")
        return
    st.rerun()


def story_tab():
    turns = store.turns(eng.story_id)
    for t in turns:
        if t["turn"] > 0:
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(md(t["action"]))
                dice_badge(t["check_json"])
        with st.chat_message("assistant", avatar=":material/auto_stories:"):
            st.markdown(md(t["narration"]))
            if t["notes"]:
                st.caption(" · ".join(md(n) for n in t["notes"]))
    if ss.pending:
        run_pending()
        return
    if not turns:
        ss.pending = {"kind": "opening"}
        st.rerun()
    out = eng.last_outcome
    if out:
        st.caption("What do you do?")
        if eng.encounter and not eng.encounter.over:
            st.caption(f"Combat, round {eng.encounter.round}: choose your move")
        for i, c in enumerate(out.choices):
            if c.combat and c.combat.startswith("attack"):
                label = f"{c.text}  ·  {c.ability} vs DEF {c.dc}"
            else:
                label = f"{c.text}  ·  {c.ability} DC {c.dc}" if c.ability and c.dc else c.text
            icon = (":material/swords:" if (c.combat or "").startswith("attack") else
                    ":material/casino:" if c.ability else ":material/arrow_forward:")
            st.button(md(label), key=f"choice_{out.turn}_{i}", on_click=queue, args=(c.text, c.model_dump()),
                      icon=icon, width="stretch")
    if action := st.chat_input("Or describe anything else you do…", submit_mode="disable"):
        queue(action)
        st.rerun()


# ============================================================================ memory graph tab
def graph_tab():
    c1, c2 = st.columns([1, 3])
    scope = c1.segmented_control("Scope", ["This tale", "Whole world"], default="This tale", key="g_scope")
    structural = c1.toggle("Show scene links", value=False, help="Scene → mentioned-entity and scene order links")
    default_types = [t for t in TYPE_STYLE if t not in ("segment", "summary", "event")]
    types = c2.pills("Node types", list(TYPE_STYLE), default=default_types, selection_mode="multi",
                     format_func=lambda t: TYPE_STYLE[t][2], key="g_types")
    highlight = tuple(eng.last_recall.included) if eng.last_recall else ()
    dark = (st.context.theme.type or "dark") == "dark"
    html, n, e = graph_html(store, eng.world_id, store.version, tuple(types or default_types),
                            eng.story_id if scope == "This tale" else None, highlight, eng.player_id, dark, structural)
    if not html:
        st.info("Nothing to show yet.", icon=":material/hub:")
        return
    st.caption(f"{n} nodes · {e} links. Size = importance · red links = hostility, green = affection · "
               "white rings = memories recalled for the latest scene · grey = dead or destroyed. Drag, zoom, hover.")
    st.iframe(html, height=660)

    st.subheader("Trace a connection")
    ents = sorted([x for x in store.nodes(eng.world_id, types=ENTITY_NODE_TYPES)], key=lambda x: -x.importance)
    if len(ents) >= 2:
        names = {x.id: f"{x.name} ({x.type})" for x in ents}
        a, b = st.columns(2)
        src = a.selectbox("From", list(names), format_func=names.get, key="path_a")
        dst = b.selectbox("To", list(names), index=1, format_func=names.get, key="path_b")
        path = eng.graph.path(eng.world_id, src, dst) if src != dst else []
        if path:
            g, nodes = eng.graph.graph(eng.world_id), store.get_nodes(path)
            parts = [f"**{md(nodes[path[0]].name)}**"]
            for u, v in zip(path, path[1:]):
                rel = "/".join(dict.fromkeys(g[u][v]["rels"])).replace("_", " ")
                parts.append(f":gray[—{md(rel)}—] **{md(nodes[v].name[:60])}**")
            st.markdown(" ".join(parts))
        elif src != dst:
            st.caption("No connection yet.")


# ============================================================================ codex tab
def codex_tab():
    types = st.pills("Filter", [t for t in ENTITY_NODE_TYPES], default=None, selection_mode="multi",
                     format_func=lambda t: TYPE_STYLE[t][2], key="codex_types")
    ents = [x for x in store.nodes(eng.world_id, types=types or ENTITY_NODE_TYPES)]
    ents.sort(key=lambda x: (-x.importance, x.name))
    if not ents:
        st.info("The codex is empty.", icon=":material/menu_book:")
        return
    nid = st.selectbox("Entry", [x.id for x in ents], key="codex_pick",
                       format_func=lambda i: next(f"{x.name} · {TYPE_STYLE[x.type][2]}" for x in ents if x.id == i))
    node = store.get_node(nid)
    titles = {s["id"]: s["title"] for s in store.list_stories(eng.world_id)}
    with st.container(border=True):
        st.subheader(md(node.name))
        row = st.container(horizontal=True)
        with row:
            st.badge(TYPE_STYLE[node.type][2], color="blue")
            if node.status:
                st.badge(node.status, color="red" if node.status.lower() in ("dead", "destroyed", "hostile") else "gray")
            for sid in store.node_story_ids(node.id):
                st.badge(titles.get(sid, f"tale {sid}"), icon=":material/auto_stories:", color="violet")
        st.markdown(md(node.description) or "_No description yet._")
        if node.aliases:
            st.caption("Also known as: " + ", ".join(md(a) for a in node.aliases))
        extra = {k: v for k, v in node.state.items() if k not in ("status",)}
        if extra:
            st.caption(" · ".join(f"{k}: {md(str(v))}" for k, v in extra.items()))
        m = st.container(horizontal=True)
        m.metric("Importance", f"{node.importance:.1f}")
        m.metric("Introduced", f"turn {node.turn}")
        m.metric("Last seen", f"turn {node.last_seen_turn}")

    edges = store.edges_of(node.id)
    others = store.get_nodes([e.dst if e.src == node.id else e.src for e in edges])
    rel_rows, event_rows = [], []
    for e in edges:
        o = others.get(e.dst if e.src == node.id else e.src)
        if not o:
            continue
        if o.type == "event" and e.rel == "involves":
            event_rows.append({"turn": o.turn, "event": o.description})
        elif e.rel not in STRUCTURAL_RELS or e.rel in ("affects",):
            rel_rows.append({"relation": e.rel.replace("_", " ") if e.src == node.id else f"← {e.rel.replace('_', ' ')}",
                             "with": o.name, "sentiment": e.sentiment, "since turn": e.turn,
                             "updated": e.updated_turn, "until": e.valid_to})
    st.markdown("**Relationships**")
    if rel_rows:
        st.dataframe(pd.DataFrame(rel_rows), hide_index=True, column_config={
            "sentiment": st.column_config.NumberColumn(format="%+d", help="-10 hatred … +10 devotion")})
    else:
        st.caption("None recorded.")
    hist = store.history(node.id)
    if hist:
        st.markdown("**How it changed over time**")
        st.dataframe(pd.DataFrame([{"turn": h["turn"], "field": h["field"], "from": h["old"], "to": h["new"],
                                    "tale": titles.get(h["story_id"], "")} for h in hist]), hide_index=True)
    if event_rows:
        st.markdown("**Appears in events**")
        st.dataframe(pd.DataFrame(sorted(event_rows, key=lambda r: r["turn"])), hide_index=True)


# ============================================================================ timeline tab
def timeline_tab():
    W, sid = eng.world_id, eng.story_id
    events = store.nodes(W, types=("event",), story_id=sid)
    names = {x.id: x.name for x in store.nodes(W, types=ENTITY_NODE_TYPES)}
    part: dict[int, list[str]] = {}
    for e in store.edges(W):
        if e.rel == "involves" and e.dst in names:
            part.setdefault(e.src, []).append(names[e.dst])
    st.subheader("Events")
    if events:
        st.dataframe(pd.DataFrame([{"turn": e.turn, "event": e.description, "importance": e.importance,
                                    "involved": part.get(e.id, [])} for e in events]),
                     hide_index=True, column_config={
                         "importance": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%d"),
                         "involved": st.column_config.ListColumn()})
    else:
        st.caption("No events yet.")
    st.subheader("Your decisions and their consequences")
    ds = store.nodes(W, types=("decision",))
    titles = {s["id"]: s["title"] for s in store.list_stories(W)}
    if ds:
        st.dataframe(pd.DataFrame([{"turn": d.turn, "tale": titles.get(d.story_id, ""), "decision": d.name,
                                    "possible consequence": d.state.get("consequence", ""),
                                    "status": d.status, "came due": d.state.get("resolved_turn"),
                                    "severity": d.state.get("severity", 0)} for d in ds]),
                     hide_index=True, column_config={
                         "severity": st.column_config.ProgressColumn(min_value=0, max_value=5, format="%d")})
    else:
        st.caption("No meaningful decisions recorded yet.")
    st.subheader("Plot threads")
    th = store.nodes(W, types=("thread",), story_id=sid)
    if th:
        st.dataframe(pd.DataFrame([{"thread": t.name, "status": t.status, "opened": t.state.get("opened_turn"),
                                    "description": t.description} for t in th]), hide_index=True)
    else:
        st.caption("No threads yet.")
    sums = [s for s in eng.summaries()]
    if sums:
        st.subheader("Chronicle")
        for s in sums:
            with st.expander(f"{s.name} · turns {s.state.get('start')}–{s.state.get('end')}"):
                st.markdown(md(s.content))


# ============================================================================ recall inspector tab
def recall_rows(items: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"used": i["included"], "memory": i["name"], "type": i["type"], "score": i["score"],
                          "relevance": i.get("rel", 0), "graph": i.get("graph", 0), "importance": i.get("imp", 0),
                          "recency": i.get("rec", 0), "why": i["why"]} for i in items])


RECALL_COLS = {
    "used": st.column_config.CheckboxColumn(help="Packed into the narrator's context"),
    "score": st.column_config.NumberColumn(format="%.2f"),
    "relevance": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f",
                                                 help="Dense + BM25 fused with reciprocal rank fusion"),
    "graph": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f",
                                             help="Personalized PageRank from the scene's seeds"),
    "importance": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
    "recency": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
    "why": st.column_config.ListColumn(help="named = mentioned by name, dense/bm25 = rank in that channel, "
                                            "graph = reached through connections"),
}


def recall_tab():
    st.caption("Before every scene the engine fuses exact-name hits, dense vectors and BM25 keywords, spreads "
               "relevance through the memory graph with Personalized PageRank, then scores by relevance, "
               "graph proximity, importance and recency.")
    turns = [t for t in store.turns(eng.story_id) if t["retrieval"].get("items")]
    if turns:
        pick = st.selectbox("Scene", [t["turn"] for t in turns][::-1], format_func=lambda x: f"Turn {x}",
                            key="recall_turn")
        t = next(t for t in turns if t["turn"] == pick)
        st.code(t["retrieval"]["query"], language=None, wrap_lines=True)
        st.dataframe(recall_rows(t["retrieval"]["items"]), hide_index=True, column_config=RECALL_COLS)

    st.subheader("Ask the memory")
    with st.form("probe"):
        q = st.text_input("Query", placeholder="Who has a grudge against me?")
        asked = st.form_submit_button("Recall", icon=":material/manage_search:")
    if asked and q.strip():
        eng.wait_ready()
        titles = {s["id"]: s["title"] for s in store.list_stories(eng.world_id)}
        res = eng.retriever.recall(eng.world_id, eng.story_id, eng.turn, q, focus_ids=[eng.player_id],
                                   story_titles=titles, player_id=eng.player_id,
                                   exclude_ids=[eng.story_node_id])
        with st.container(border=True):
            st.markdown(md(res.context))
        st.dataframe(recall_rows(res.to_log()["items"]), hide_index=True, column_config=RECALL_COLS)


# ============================================================================ director tab
def director_tab():
    arc = eng.arc
    if arc:
        with st.container(border=True):
            st.markdown(f"**Premise.** {md(arc.premise)}")
            m = st.container(horizontal=True)
            m.metric("Act", arc.current_act)
            m.metric("Tension", f"{arc.tension}/10")
            st.markdown(f"**Act goal.** {md(arc.act_goal)}")
            st.markdown("**Next beats**\n" + "\n".join(f"{i}. {md(b)}" for i, b in enumerate(arc.next_beats, 1)))
            if arc.callbacks:
                st.markdown("**Callbacks:** " + ", ".join(md(c) for c in arc.callbacks))
            if arc.consequences_to_payoff:
                st.markdown("**Consequences to land:** " + "; ".join(md(c) for c in arc.consequences_to_payoff))
            if arc.foreshadowing:
                st.markdown(f"**Foreshadowing:** {md(arc.foreshadowing)}")
    if st.button("Re-plan now", icon=":material/refresh:"):
        with st.spinner("The Director reconsiders…"):
            eng.wait_ready()
            eng.replan()
        st.rerun()

    st.markdown("**Consequence clocks**")
    st.caption("Each meaningful decision starts a clock; scenes, failed rolls and time tick it. When a clock "
               "fills, its payoff is forced into the next scene (Blades in the Dark-style progress clocks).")
    all_clocks = store.nodes(eng.world_id, types=("clock",))
    if all_clocks:
        st.dataframe(pd.DataFrame([{
            "clock": k.name, "type": "front" if k.state.get("front") else "consequence",
            "progress": int(k.state.get("filled", 0)) / max(1, int(k.state.get("segments", 4))),
            "segments": f"{k.state.get('filled', 0)}/{k.state.get('segments', 4)}",
            "status": k.state.get("status", ""), "payoff": k.state.get("payoff", ""),
            "started": k.turn, "fired": k.state.get("fired_turn")} for k in all_clocks]),
            hide_index=True, column_config={
                "progress": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})
    else:
        st.caption("No clocks yet.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Dormant characters (callback candidates)**")
        for d in eng.dormant_entities():
            st.markdown(f"- {md(d.name)} · last seen turn {d.last_seen_turn}")
    with right:
        stats = store.stats(eng.world_id)
        st.markdown("**Memory by type**")
        st.bar_chart(pd.DataFrame({"count": stats["nodes"]}), horizontal=True, height=260)
        st.markdown("**Agent activity this session**")
        rows = [{"agent": k, "calls": v["calls"], "total s": round(v["seconds"], 1),
                 "avg s": round(v["seconds"] / max(1, v["calls"]), 1)} for k, v in eng.llm.stats.by_agent.items()]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)


# ============================================================================ play
def play():
    if eng.story_id is None:
        ss.view = "home"
        st.rerun()
    sidebar()
    tabs = st.tabs([":material/auto_stories: Story", ":material/hub: Memory graph", ":material/menu_book: Codex",
                    ":material/timeline: Timeline", ":material/manage_search: Recall inspector",
                    ":material/theater_comedy: Director"], key="tabs", on_change="rerun")
    renderers = [story_tab, graph_tab, codex_tab, timeline_tab, recall_tab, director_tab]
    for i, (tab, render) in enumerate(zip(tabs, renderers)):
        if tab.open or (i == 0 and ss.pending):
            with tab:
                render()


if ss.view == "play":
    play()
else:
    home()
