# Loreweaver — an AI Dungeon Master that remembers

**M1-P8: AI-based interactive storytelling agent.** The user gives a theme, and a local LLM runs a
D&D-style adventure that keeps evolving. Every character, place, event, decision and consequence is
stored in a **temporal knowledge graph** and recalled through **hybrid GraphRAG**. Contradictions are
caught and repaired before they can enter memory. Choices start **consequence clocks** that come due
on schedule. Fights run through a **deterministic encounter engine**. Stories told in the same world
share one connected memory.

Everything runs locally on an 8 GB GPU (`qwen3.5:4b` + `qwen3-embedding:0.6b` through Ollama).

## How it meets the problem statement

| Requirement | Implementation |
|---|---|
| User provides a theme | **Worldbuilder** agent turns the theme into a world bible (setting, conflict, locations, factions, NPCs with goals and secrets, opening hook, and a campaign **front** with grim portents), seeded into the graph |
| Continuously evolving, autonomous story | **Director** agent keeps a 3-act arc (tension, next beats, callbacks to dormant characters, clocks to build toward) and steers the **Narrator**. The front advances as time passes, so the world moves even when the player doesn't |
| Remembers characters and events | **Chronicler** extracts entities, state changes, relations, events, threads, decisions and clock ticks after every scene. Each fact must be **grounded in the scene text** before it enters memory. **Hybrid retriever** (exact names + dense + BM25 + relation channel → RRF → Personalized PageRank) brings the relevant memories back. An authoritative **ledger** of current state closes every prompt |
| Later segments change with user decisions | Every meaningful decision starts a **consequence clock** with a concrete payoff. Scenes, failed rolls and time tick it. When it fills, **code** makes the payoff mandatory in the next scene and links `decision → clock → event`. NPC attitudes accumulate on edges. Fights the hero starts or walks into are resolved round by round by the rules |

## Architecture

```
 Streamlit UI  (Story · Memory graph · Codex · Timeline · Recall inspector · Director)
        │
   GameEngine ──► Worldbuilder · Adjudicator · Narrator · Chronicler · Director · Summarizer · Continuity
        │              (one local LLM, seven roles; structured output is grammar-constrained)
        ├──► Rules (d20 checks, sheet)   Encounter engine (combat phase)   Clocks (consequences, fronts)
        ▼
   MemoryRetriever (hybrid + graph) ◄── MemoryStore (SQLite: nodes, edges, history, FTS5, vectors) + StoryGraph
```

**One turn:**
1. **Action:** an offered choice or free text. If the hero attacks, or a fight is running, the **encounter engine** resolves the round in code (attack vs DEF, damage, enemy turns, morale). Otherwise the Adjudicator picks an ability check and code rolls it. Failed rolls tick the relevant clock. The **action precheck** flags false premises (a dead character addressed, an item the hero doesn't carry).
2. **Prompt:** memory recall, Director notes (plus "MANDATORY THIS SCENE" when a clock is due) and the **ledger** go to the Narrator, which streams the scene. A live filter drops copied paragraphs.
3. **Background, while you read:**
   1. If the scene carries a risk signal (a dead character acts, the precheck fired, the hero uses something), the **continuity checker** compares it with the ledger and repairs contradicting sentences.
   2. The Chronicler extracts facts from the *corrected* scene.
   3. Grounding and canon guards filter them, and the graph is updated. A relation that contradicts an older one ends it (`valid_to`).
   4. Clocks tick, fire or are averted; a fight that broke out starts an encounter.
   5. Changed nodes are re-embedded, a chapter is summarized every 5 turns, and the Director re-plans every 4 turns.
   6. The stable part of the next narrator prompt is sent to Ollama to **warm the KV cache**, so the next scene starts sooner.

## Reliability layers (phase 2)

Each layer answers a limitation found in phase 1; [PLAN.md](PLAN.md) §7–10 has the research and the real-run lessons.

| Layer | What it does | Why |
|---|---|---|
| Grammar-constrained JSON | Structured agents use Ollama `/api/generate` with `format=<schema>` | `/api/chat` ignores `format` when thinking is off on qwen3.5 (ollama#14645, verified locally). **0 fallbacks in 61 structured calls** across two 12-turn runs |
| Ledger | Hero HP and *exactly* what they carry, who is present, who is elsewhere, who is dead, the combat state, threats about to fire; placed last in the prompt | NCP-Bench: missing explicit state tracking is the main cause of narrative contradictions |
| Evidence grounding | An extracted entity, relation, event participant, item or combatant is kept only if the scene text mentions it | A hallucinated fact in memory spreads to every later prompt |
| Action precheck | Before narration, code finds false premises in the player's action (a dead character addressed, "my <item>" not carried) and gives the narrator a binding RULES CHECK | The adversarial eval showed the narrator follows the player's premise whatever the ledger says; this cut phantom-item contradictions from 75% to 0–25% |
| Canon guards | No resurrection unless the scene justifies it; used items must be carried; combatants' fates and HP belong to the combat engine | — |
| Continuity checker + repair | The LLM flags `dead_acting` and `missing_item` contradictions with an **exact quote**. A quote not found in the scene, or a flag that fails a deterministic check of its kind, is discarded. Confirmed sentences are replaced in place *before* extraction; a scene built on the contradiction (2+ detector hits) is rewritten once | ConStory-Checker-style evidence grounding. Precision was tuned after the checker was seen "correcting" legitimate in-scene changes |
| Consequence clocks | 4/6/8-segment clocks with concrete payoffs; cooldowns; fronts with portents that cannot fire before turn 18 | Blades in the Dark progress clocks and Dungeon World fronts: delayed consequences that are concrete and can't resolve early |
| Encounter engine | Tiered stat blocks under a level-based difficulty budget; attack/defend/potion/flee/parley; morale; code writes who died, fled or surrendered | Setting the DC (NeurIPS 2025) and tool-grounded rules: the LLM narrates results, it never decides them |

## Efficiency and memory quality (phase 3)

Phase 3 profiled real turns and tuned for speed and recall; [PLAN.md](PLAN.md) §11–13 has the measurements and research.

| Change | What it does | Result |
|---|---|---|
| Compact structured output | Structured calls ask for minified one-line JSON that omits empty or unchanged fields; the grammar still guarantees structure | Chronicler A/B on 5 real scenes: **−38% tokens, −37% decode time**, same validity |
| Prefix KV-cache warm-up | The narrator prompt puts its stable part first (system, world, story so far, recent scenes); after each commit that prefix is pre-computed while the player reads | Time to first word on choice turns **4.5 s → 2.3 s** |
| Risk-gated continuity check | The checker runs only when a dead character acts, the precheck fired, or the hero uses something (ConStory-Bench: verify on risk signals) | Fires on 0 of 27 ordinary scenes, so ordinary turns skip the extra LLM call |
| Query-gated PageRank | Edge weights into a node are scaled by that node's similarity to the query (with a floor), so the graph walk flows toward relevant memories | Hit@5 **81.5% → 83.6%**, MRR 0.580 → 0.590 over 3 stories; now the default |
| Temporal edge invalidation | A new relation that contradicts an old one between the same pair (sentiment sign flip) closes the old edge with `valid_to` (Zep/Graphiti style) | Cards and the graph walk use current facts; the Codex shows when a relation ended |

Each change can be switched off for ablation: `compact_json`, `warm_prefix`, `gate_checker` and `graph_mode` (`ppr` · `gated_ppr` · `spread`) in `loreweaver/config.py`.

### The memory system

* **One graph for everything:** node types `player, character, creature, location, item, faction, lore, event, thread, decision, clock, segment, summary, story`. Relation edges carry a sentiment from −10 to +10. Structural edges (`mentions`, `involves`, `occurred_at`, `affects`, `started`, `threatens`, `led_to`, `advances`, `appears_in`, `next`) tie scenes, events, choices and consequences together.
* **Temporal:** every state change is versioned per turn ("Silas: alive → dead @ T12"). Life state is kept separate from attitude. A contradicted relation keeps its history but is marked ended (`valid_to`).
* **Interconnected stories:** worlds own nodes. A new tale in an existing world inherits its history, and active clocks carry over.
* **Hybrid retrieval:** exact names, dense, BM25 and edge-relation channels are fused by RRF and spread with query-gated Personalized PageRank (√degree-normalized). Results are scored on relevance, graph, importance and recency, with boosts for open threads and clocks near firing, then packed into sections within a token budget.
* **Entity resolution:** exact/alias → distinctive token subset → fuzzy → embedding. Generic words never merge, and numbered individuals ("Cult Guard 1/2") never merge.

## Setup

```bash
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:0.6b
pip install -r requirements.txt
streamlit run app.py          # web UI
python cli.py                 # terminal client (/sheet /clocks /memory)
```

Settings in `loreweaver/config.py` can be overridden with `LOREWEAVER_*` environment variables (model, context size, `use_ledger`, `continuity_check`, `front_min_turn`…). Saves go to `data/loreweaver.db`.

**Hardware notes (RTX 5060 Laptop, 8 GB):**
- Both models stay resident (about 7.2 GB).
- The first words of a scene appear after about 2–3 s (warm prefix), and the full scene streams in about 8–12 s.
- The background commit (check, extract, write, warm-up) takes about 15–30 s while you read.

## Using the UI

* **Story:** streamed scenes, dice badges, what memory recorded and repaired after each scene ("🛠 Continuity fix"), choice buttons (the combat menu during fights) and free text.
* **Sidebar:** character sheet, combat panel (enemy HP, round, your AC), threat clocks with progress bars, memory status.
* **Memory graph:** an interactive graph of this tale or the whole world, including clocks. *Trace a connection* shows how two entities are linked.
* **Codex:** each entity's card, relationships, state history and appearances across tales.
* **Timeline:** events, your decisions and whether their consequences came due, plot threads, the chronicle.
* **Recall inspector:** what was retrieved for each scene and why. *Ask the memory* lets you query it.
* **Director:** the arc plan, the clocks table (front and consequences), dormant characters, agent statistics.

## Evaluation

### Memory recall: `eval/recall_eval.py`
Probe questions are generated from a played story's graph, covering only memories at least 3 turns old, and scored under five retrieval configurations. **Results on a 15-turn held-out story** (80 probes):

| configuration | name | description | event | relationship | **Hit@5** | Hit@10 | MRR | **packed** |
|---|---|---|---|---|---|---|---|---|
| **full (named + dense + BM25 + relation + graph)** | 100% | 90% | 86% | 86% | **90%** | **98%** | **0.59** | **100%** |
| no relation channel | 100% | 90% | 86% | 14% | 78% | 88% | 0.50 | 100% |
| no graph | 100% | 90% | 50% | 86% | 80% | 96% | 0.57 | 99% |
| dense + BM25 (classic hybrid RAG) | 100% | 95% | 91% | 14% | 80% | 84% | 0.50 | 85% |
| dense only (plain vector RAG) | 87% | 90% | 86% | 0% | 71% | 80% | 0.39 | 80% |
| BM25 only | 87% | 70% | 77% | 21% | 69% | 82% | 0.37 | 85% |

*Packed* means the memory actually reached the narrator's prompt.
- **Graph expansion** drives multi-hop event recall (86% vs 50% without it).
- **The relation channel** drives relationship recall (86% vs 14%).

Methodology: √degree normalization and multi-answer probes were tuned on an 8-turn development story. The relation channel was added after the first held-out run, so its held-out number is optimistic.

### Consistency under adversarial play: `eval/consistency_eval.py`
In the spirit of NCP-Bench's adversarial player, each probe tries to make the narrator contradict canon, run on fresh copies of a played story:
- ask a dead character for help
- use an item the hero doesn't have
- greet a hostile NPC as an old friend
- claim a false promise

A separately-worded judge must quote evidence for any "yes".

**Results** (12-turn story, 4 probes × 4 conditions × 4 reps = 64 real-model runs; % of scenes that contradicted canon):

| probe | baseline (phase 1) | + ledger | + ledger + precheck | full (+ checker/repair) |
|---|---|---|---|---|
| ask a dead character for help | 0% | 50% | **0%** | **0%** |
| use an item the hero lacks | 75% | 75% | 25% | **0%** |
| greet a hostile NPC as a friend | 0% | 0% | 0% | 0% |
| claim a false promise | 25% | 0% | 0% | 50%* |
| **all probes** | 25% | 31% | **6%** | 12% |
| avg turn latency | 15.0 s | 14.2 s | 15.4 s | 18.9 s |

What this does and does not show:
* **The action precheck works.** The narrator follows a false premise in the player's action whatever the ledger says ("seduced by the narrative"). Correcting that premise in code *before* narration took phantom-item contradictions from 6/8 to 1/8 (Fisher p = 0.04), and the checker caught the remaining case. Across all probes the precheck conditions cut contradictions from 9/32 to 3/32 (p = 0.11: consistent direction, not significant at this sample size).
* **A ledger alone is not enough.** It did not reliably stop false premises, which matches the literature: state injection helps, but external rule checks help more.
* \*Both "false promise" hits under *full* were judge errors on inspection. In one the hero restates the claim; in the other the NPC says the money "was spent" without confirming any promise. The judge is the same 4B model (constrained to quote evidence), and cells of 4 reps are noisy: an earlier 3-rep run showed the dead-character baseline at 67%. Treat single cells as indicative and the aggregate trend as the result.
* An earlier version that relied only on post-hoc repair scored **50%** overall. That result is why the precheck exists (see PLAN.md §10).

**Recall on a second, untuned story** (the same 12-turn run, 55 probes): full system Hit@5 **80%** vs 66% for dense + BM25 and 58% for plain vector RAG; Hit@10 95%; MRR 0.58 vs 0.47 / 0.40; 100% of targets reached the prompt. Event recall was weaker in this story (44% Hit@5).

### Phase 3 retrieval ablation: `eval/recall_eval.py`
Three played stories (the 8-turn dev story, the 15-turn held-out story and the 12-turn phase-2 story), 195 probes:

| configuration | Hit@5 | Hit@10 | MRR | packed |
|---|---|---|---|---|
| phase 2 (named + dense + BM25 + relation, PPR) | 81.5% | 96.4% | 0.580 | 99.5% |
| **query-gated PPR (new default)** | **83.6%** | 95.9% | **0.590** | 99.5% |
| spreading activation (T=3, α=0.7) | 84.6% | 94.9% | 0.559 | 99.5% |
| + fact seeds (HippoRAG-2 triples as walk seeds) + gated PPR | 84.6% | 95.9% | 0.586 | 99.5% |
| + facts as score boosts | 72.8% | 92.3% | 0.525 | 99.5% |
| classic hybrid (dense + BM25) | 73.8% | 86.2% | 0.475 | 88.2% |

- **Gated PPR** improved Hit@5 and MRR on all three stories at negligible cost.
- **Spreading activation** gets the target into the top 5 more often but ranks it lower, so MRR drops.
- **Facts as boosts are a negative result:** they flood the ranking with generic relation partners. Facts as walk seeds are neutral within noise and stay opt-in (`"fact_seeds"` channel).

### Latency: `eval/latency_eval.py`
Plays the same choice turns from fresh copies of a saved story with phase-2 settings (pretty JSON, no warm-up, checker on every scene) and phase-3 settings, and reports time to first token, narration time and background commit time. Time to first token on choice turns fell from **4.5 s to 2.3 s**.

```bash
python eval/latency_eval.py --db data/loreweaver.db --story 1 --turns 4
```

### Final smoke run
A 6-turn autoplay with the final settings (`python cli.py --auto 6 --theme "…"`) on the RTX 5060:
- **Calls:** 43 LLM calls, 0 constrained-JSON fallbacks in 10 structured calls.
- **Memory:** 2 decisions started clocks, and 3 clocks were ticking at the end. The graph held 18 events, 6 locations and 201 edges.
- **Continuity:** 0 repairs were needed.

## Tests

```bash
python -m pytest -q
```

44 tests, no GPU needed. They cover:
- **rules and parsing:** dice, choice parsing, JSON drift tolerance, the constrained-JSON path and its fallback
- **memory:** the temporal store, FTS, entity resolution (including numbered individuals), graph recall of unmentioned connected memories, relation-aware recall, cross-story labelling, a contradicting relation superseding the old one (`valid_to`)
- **efficiency:** the warmed prefix is an exact prefix of the next narrator prompt (so the KV cache is reused); the checker runs only on risk signals
- **grounding and guards:** evidence grounding, the resurrection and phantom-item guards
- **continuity:** the checker's quote evidence, kind verification and in-place repair
- **clocks:** ticks, cooldowns, portents, the front-fires-late rule, firing and averting
- **combat:** stat blocks, rounds to completion, defend/potion/flee/parley, intent detection, difficulty budget
- **end-to-end engine runs:**
  - a decision's clock is forced into a later scene and then fires
  - a fight is resolved by code and written to memory
  - a hero-initiated attack starts an encounter before narration
  - the ledger, checker and guards together keep the dead dead

## Project layout

```
loreweaver/
  config.py  llm.py  schemas.py  prompts.py  rules.py  viz.py
  engine.py          turn pipeline, ledger, background commit, encounters, clocks, summaries, re-planning
  clocks.py          consequence clocks and fronts
  combat.py          encounter engine
  memory/  store.py graph.py resolver.py retriever.py
  agents/  worldbuilder.py adjudicator.py narrator.py chronicler.py director.py summarizer.py continuity.py
app.py · cli.py · eval/recall_eval.py · eval/consistency_eval.py · eval/latency_eval.py · tests/ · PLAN.md
```

## Limitations

* The continuity checker auto-repairs only two precise kinds of contradiction. Subtler drift (a minor detail, a character's tone) is reduced by the ledger but not caught.
* Clock payoffs and encounter tiers come from a 4B model. Code bounds their pacing and difficulty, but their quality varies.
* Combat is theatre-of-the-mind: no positioning or zones, a single hero, at most four enemies.
* Evaluation uses the same 4B model as judge, constrained by quote evidence. A larger independent judge would be stronger.
