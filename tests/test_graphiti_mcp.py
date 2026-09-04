"""Knowledge-graph memory: agent-authored facts with a timeline.

The premise of this service is that the *agent* decides what is stored, how it
is typed, and when it stopped being true — no extraction pass, no LLM in the
write path. These tests pin that: every fact here is constructed by hand, with
schema invented on the spot, and nothing configures a model.
"""

import math
import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest

GRAPH_HOST = os.environ.get("GRAPH_HOST", "localhost")
GRAPH_PORT = int(os.environ.get("GRAPH_PORT", "6379"))


def _falkordb_available() -> bool:
    try:
        import falkordb
        falkordb.FalkorDB(host=GRAPH_HOST, port=GRAPH_PORT).list_graphs()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _falkordb_available(),
    reason=f"FalkorDB not reachable at {GRAPH_HOST}:{GRAPH_PORT}")


@pytest.fixture
def graph(monkeypatch):
    """A private graph per test, so ordering and leftovers cannot matter."""
    import graphiti_mcp.store as store
    import graphiti_mcp.embed as embed
    import graphiti_mcp.server as server

    monkeypatch.setattr(store, "GRAPH_HOST", GRAPH_HOST)
    monkeypatch.setattr(store, "GRAPH_PORT", GRAPH_PORT)
    monkeypatch.setattr(store, "GRAPH_DATABASE", f"test_{uuid.uuid4().hex[:12]}")
    monkeypatch.setattr(store, "GROUP_ID", "test")
    monkeypatch.setattr(store, "_driver", None)
    monkeypatch.setattr(store, "_indices_ready", False)
    monkeypatch.setattr(embed, "_client", None)
    monkeypatch.setattr(embed, "_checked", True)      # no embedder by default
    yield server


def _fact(server, subject, relation, obj, text, s_type="Person", o_type="Topic"):
    return server.Fact(subject=subject, subject_type=s_type, relation=relation,
                       object=obj, object_type=o_type, fact=text)


# ── Writing ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_fact_keeps_its_sentence(graph):
    """The triple is an index over the sentence, not a replacement for it —
    nothing the agent wrote should be reduced to a schema."""
    r = await graph.save_facts(facts=[_fact(
        graph, "Max", "PREFERS", "commit granularity",
        "Max prefers function-level commit granularity")])
    stored = r["facts"][0]
    assert stored["fact"] == "Max prefers function-level commit granularity"
    assert (stored["subject"], stored["relation"], stored["object"]) == (
        "Max", "PREFERS", "commit granularity")


@pytest.mark.asyncio
async def test_schema_is_invented_not_declared(graph):
    """Entity types and relation names are free-form: a vocabulary the agent
    makes up mid-conversation costs no migration."""
    await graph.save_facts(facts=[
        _fact(graph, "Nanhi", "RUNS_ON", "kimi-k3", "Nanhi runs on kimi-k3",
              s_type="Agent", o_type="Model"),
        _fact(graph, "Max", "COMMITTED_TO", "design review",
              "Max will review the design", o_type="Commitment"),
    ])
    vocab = await graph.list_vocabulary()
    assert {"Agent", "Model", "Commitment"} <= set(vocab["entity_types"])
    assert {"RUNS_ON", "COMMITTED_TO"} <= set(vocab["relations"])


@pytest.mark.asyncio
async def test_the_same_entity_is_reused_not_duplicated(graph):
    """Two facts about Max must attach to one Max, or the graph fragments and
    `get_entity` only ever sees half of what is known."""
    await graph.save_facts(facts=[_fact(graph, "Max", "PREFERS", "a", "Max prefers a")])
    await graph.save_facts(facts=[_fact(graph, "Max", "DISLIKES", "b", "Max dislikes b")])
    entity = await graph.get_entity("Max")
    assert entity["found"] and len(entity["facts"]) == 2


@pytest.mark.asyncio
async def test_an_existing_entity_gains_types_without_losing_them(graph):
    await graph.save_facts(facts=[_fact(graph, "Max", "IS", "engineer",
                                        "Max is an engineer", s_type="Person")])
    await graph.save_facts(facts=[_fact(graph, "Max", "REVIEWS", "PRs",
                                        "Max reviews PRs", s_type="Reviewer")])
    entity = await graph.get_entity("Max")
    assert {"Person", "Reviewer"} <= set(entity["types"])


# ── Stories ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_one_story_is_shared_by_a_batch(graph):
    """Facts from one conversation share a story rather than each carrying a
    copy — otherwise a four-hit search returns the same narrative four times."""
    r = await graph.save_facts(
        facts=[_fact(graph, "Max", "PREFERS", "a", "Max prefers a"),
               _fact(graph, "Max", "PREFERS", "b", "Max prefers b")],
        story="## How this came up\nWe were designing the memory service.")
    assert r["episode_id"]
    story = await graph.get_story(r["episode_id"])
    assert story["fact_count"] == 2
    assert "designing the memory service" in story["story"]
    assert {f["episode_id"] for f in r["facts"]} == {r["episode_id"]}


@pytest.mark.asyncio
async def test_a_later_fact_can_join_an_existing_story(graph):
    """Adding to something already recorded should link, not re-narrate."""
    first = await graph.save_facts(
        facts=[_fact(graph, "Max", "PREFERS", "a", "Max prefers a")],
        story="The original conversation.")
    await graph.save_facts(
        facts=[_fact(graph, "Max", "PREFERS", "b", "Max prefers b")],
        episode_id=first["episode_id"])
    story = await graph.get_story(first["episode_id"])
    assert story["fact_count"] == 2


@pytest.mark.asyncio
async def test_facts_without_a_story_are_still_valid(graph):
    r = await graph.save_facts(facts=[_fact(graph, "Max", "PREFERS", "a",
                                            "Max prefers a")])
    assert r["episode_id"] is None and r["facts"][0]["fact_id"]


# ── Time ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_superseding_keeps_the_old_fact(graph):
    """The reason for a graph over a key-value store: what was once true stays
    answerable."""
    r = await graph.save_facts(facts=[_fact(graph, "Nanhi", "RUNS_ON", "glm",
                                            "Nanhi runs on glm")])
    fid = r["facts"][0]["fact_id"]
    done = await graph.supersede_fact(fid, reason="switched endpoints")
    assert done["superseded"] and done["invalid_at"]
    assert done["fact"] == "Nanhi runs on glm"          # still there, verbatim

    entity = await graph.get_entity("Nanhi")
    assert entity["facts"][0]["superseded"] is True


@pytest.mark.asyncio
async def test_a_backdated_fact_keeps_the_date_it_was_given(graph):
    """Recording something learned late must not claim it started now."""
    when = datetime.now(timezone.utc) - timedelta(days=900)
    r = await graph.save_facts(
        facts=[_fact(graph, "Max", "WORKS_AT", "somewhere", "Max works there")],
        valid_at=when)
    assert r["facts"][0]["valid_at"].startswith(when.date().isoformat())


# ── Retrieval ────────────────────────────────────────────


@pytest.fixture
def embedded(graph, monkeypatch):
    """A deterministic stand-in embedder, so retrieval is testable offline."""
    import graphiti_mcp.embed as embed

    def vector(text: str) -> list[float]:
        seed = sum(ord(c) for c in text.lower() if c.isalnum()) or 1
        v = [math.sin(seed * (i + 1)) for i in range(8)]
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v]

    async def fake(text):
        return vector(text) if text else None

    monkeypatch.setattr(embed, "embed", fake)
    monkeypatch.setattr(graph.embed, "embed", fake)
    return graph


@pytest.mark.asyncio
async def test_search_finds_a_fact_by_its_sentence(embedded):
    await embedded.save_facts(facts=[_fact(
        embedded, "Max", "PREFERS", "commit granularity",
        "Max prefers function-level commit granularity")])
    hits = await embedded.search_facts("Max prefers function-level commit granularity")
    assert hits and hits[0]["subject"] == "Max"
    assert hits[0]["fact_id"]


@pytest.mark.asyncio
async def test_search_hides_superseded_facts_by_default(embedded):
    """Current picture by default; history only when asked for, so a stale
    fact cannot quietly present itself as true."""
    text = "Nanhi runs on glm"
    r = await embedded.save_facts(facts=[_fact(embedded, "Nanhi", "RUNS_ON", "glm", text)])
    await embedded.supersede_fact(r["facts"][0]["fact_id"])

    assert await embedded.search_facts(text) == []
    history = await embedded.search_facts(text, include_superseded=True)
    assert history and history[0]["superseded"] is True


@pytest.mark.asyncio
async def test_search_returns_the_episode_id_so_linking_is_cheap(embedded):
    """Sharing a story has to be easier than rewriting one, or duplicates win
    on convenience alone."""
    text = "Max prefers function-level commits"
    await embedded.save_facts(facts=[_fact(embedded, "Max", "PREFERS", "commits", text)],
                              story="Where this came from.")
    hits = await embedded.search_facts(text)
    assert hits[0]["episode_id"]
    story = await embedded.get_story(hits[0]["episode_id"])
    assert "Where this came from" in story["story"]


@pytest.mark.asyncio
async def test_writes_still_work_with_no_embedding_service(graph):
    """An unreachable embedding endpoint must not cost the write — the fact is
    worth keeping without a vector and can be embedded later."""
    r = await graph.save_facts(facts=[_fact(graph, "Max", "PREFERS", "a",
                                            "Max prefers a")])
    assert r["facts"][0]["fact_id"]
    assert await graph.search_facts("anything") == []      # degrades, not errors
    assert (await graph.get_entity("Max"))["found"] is True


@pytest.mark.asyncio
async def test_unknown_entity_reports_not_found(graph):
    assert (await graph.get_entity("Nobody"))["found"] is False


@pytest.mark.asyncio
async def test_keyword_search_works_without_an_embedder(graph):
    """The reason this runs on FalkorDB rather than an embedded backend:
    fulltext indices are real here, so an exact name is findable even with no
    embedding service configured — the case vectors are worst at."""
    await graph.save_facts(facts=[
        _fact(graph, "Heather", "RUNS_ON", "glm-5.3", "Heather runs on glm-5.3",
              s_type="Agent", o_type="Model"),
        _fact(graph, "Nanhi", "RUNS_ON", "kimi-k3", "Nanhi runs on kimi-k3",
              s_type="Agent", o_type="Model"),
    ])
    hits = await graph.search_facts("Heather")
    assert [h["subject"] for h in hits] == ["Heather"]


@pytest.mark.asyncio
async def test_search_merges_keyword_and_semantic_hits(embedded):
    """Each mode misses what the other catches: a paraphrase defeats keywords,
    an unseen proper noun defeats vectors. Results are the union, deduped."""
    await graph_save(embedded)
    hits = await embedded.search_facts("Heather", limit=10)
    assert any(h["subject"] == "Heather" for h in hits)
    ids = [h["fact_id"] for h in hits]
    assert len(ids) == len(set(ids)), "a fact matched by both modes was returned twice"


async def graph_save(server):
    await server.save_facts(facts=[
        _fact(server, "Heather", "RUNS_ON", "glm-5.3", "Heather runs on glm-5.3",
              s_type="Agent", o_type="Model")])
