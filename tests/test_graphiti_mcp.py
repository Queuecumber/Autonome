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

import falkordb
import pytest
from redis.exceptions import RedisError

GRAPH_HOST = os.environ.get("GRAPH_HOST", "localhost")
GRAPH_PORT = int(os.environ.get("GRAPH_PORT", "6379"))


@pytest.fixture(scope="session")
def graph_backend() -> None:
    """Require FalkorDB in CI while allowing local runs without the service.

    Returns:
        None when the configured database is reachable.

    Raises:
        pytest.fail.Exception: If REQUIRE_GRAPH_TESTS=1 and FalkorDB is unavailable.
        pytest.skip.Exception: If FalkorDB is unavailable during an optional run.
    """
    try:
        falkordb.FalkorDB(host=GRAPH_HOST, port=GRAPH_PORT,
                         socket_connect_timeout=2, socket_timeout=2).list_graphs()
    except RedisError as error:
        message = f"FalkorDB not reachable at {GRAPH_HOST}:{GRAPH_PORT}: {error}"
        if os.environ.get("REQUIRE_GRAPH_TESTS") == "1":
            pytest.fail(message, pytrace=False)
        pytest.skip(message)


@pytest.fixture
def graph(monkeypatch, graph_backend):
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


@pytest.mark.asyncio
@pytest.mark.parametrize("subject_type,object_type", [
    ("Chat room", "Character artifact"),
    ("  Chat\t room  ", " Character\nartifact "),
])
async def test_multiword_entity_types_round_trip(graph, subject_type, object_type):
    """Subject and object types accept whitespace and return canonical labels."""
    result = await graph.save_facts(facts=[
        _fact(graph, "Workshop chat", "CONTAINS", "Copper sigil",
              "Workshop chat contains the Copper sigil", subject_type, object_type)])

    assert result["facts"][0]["subject"] == "Workshop chat"
    assert result["facts"][0]["object"] == "Copper sigil"
    assert (await graph.get_entity("Workshop chat"))["types"] == ["Chat_room"]
    assert (await graph.get_entity("Copper sigil"))["types"] == ["Character_artifact"]
    assert (await graph.list_vocabulary())["entity_types"] == [
        "Character_artifact", "Chat_room"]


@pytest.mark.asyncio
async def test_multiword_entity_types_reuse_existing_labels(graph):
    """Adding a multi-word type preserves old types and reuses underscored aliases."""
    for subject_type, object_type in [
        ("Place", "Character_artifact"),
        ("Chat room", "Character artifact"),
        ("Chat_room", "Character_artifact"),
    ]:
        await graph.save_facts(facts=[
            _fact(graph, "Workshop chat", "CONTAINS", "Copper sigil",
                  "Workshop chat contains the Copper sigil", subject_type, object_type)])

    room = await graph.get_entity("Workshop chat")
    artifact = await graph.get_entity("Copper sigil")
    assert sorted(room["types"]) == ["Chat_room", "Place"]
    assert artifact["types"] == ["Character_artifact"]
    assert len(room["facts"]) == len(artifact["facts"]) == 3
    assert (await graph.list_vocabulary())["entity_types"] == [
        "Character_artifact", "Chat_room", "Place"]


@pytest.mark.asyncio
async def test_whitespace_only_entity_types_are_omitted(graph):
    """Whitespace-only types behave like the optional empty type."""
    await graph.save_facts(facts=[
        _fact(graph, "Workshop chat", "CONTAINS", "Copper sigil",
              "Workshop chat contains the Copper sigil", " \t ", "\n")])

    assert (await graph.get_entity("Workshop chat"))["types"] == []
    assert (await graph.get_entity("Copper sigil"))["types"] == []
    assert (await graph.list_vocabulary())["entity_types"] == []


@pytest.mark.asyncio
async def test_entity_types_still_reject_punctuation(graph):
    """Whitespace normalization must retain validation of other label characters."""
    with pytest.raises(ValueError, match="node_labels"):
        await graph.save_facts(facts=[
            _fact(graph, "Workshop chat", "CONTAINS", "Copper sigil",
                  "Workshop chat contains the Copper sigil", "Chat` room", "Artifact")])


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


@pytest.mark.asyncio
async def test_source_only_batches_keep_shared_provenance(graph):
    """Attribution survives saving and retrieval even without narrative text."""
    source = "archive/2026-05-12.md"
    result = await graph.save_facts(
        facts=[_fact(graph, "Max", "PREFERS", "a", "Max prefers a"),
               _fact(graph, "Max", "PREFERS", "b", "Max prefers b")],
        source=source)
    assert result["episode_id"]
    assert {fact["episode_id"] for fact in result["facts"]} == {result["episode_id"]}
    story = await graph.get_story(result["episode_id"])
    assert story["source"] == source
    assert story["story"] == ""
    assert story["fact_count"] == 2
    hits = await graph.search_facts("Max")
    assert len(hits) == 2
    assert {hit["episode_id"] for hit in hits} == {result["episode_id"]}


@pytest.mark.asyncio
async def test_story_keeps_narrative_source_and_dates_separate(graph):
    """A historical source date must not be reported as the date it was saved."""
    when = datetime(2026, 5, 12, tzinfo=timezone.utc)
    result = await graph.save_facts(
        facts=[_fact(graph, "Max", "PREFERS", "a", "Max prefers a")],
        story="We discussed this in May.", source="archive/2026-05-12.md",
        valid_at=when)
    story = await graph.get_story(result["episode_id"])
    assert story["story"] == "We discussed this in May."
    assert story["source"] == "archive/2026-05-12.md"
    assert datetime.fromisoformat(story["valid_at"]) == when
    assert datetime.fromisoformat(story["recorded_at"]) > when


@pytest.mark.asyncio
@pytest.mark.parametrize("provenance", [{"story": "ignored narrative"},
                                       {"source": "ignored source"}])
async def test_reusing_an_episode_rejects_conflicting_provenance(graph, provenance):
    """New attribution cannot be silently discarded when an episode is reused."""
    with pytest.raises(ValueError, match="episode_id"):
        await graph.save_facts(
            facts=[_fact(graph, "Max", "PREFERS", "a", "Max prefers a")],
            episode_id="existing-episode", **provenance)
    assert await graph.list_facts() == {"facts": [], "next_cursor": None}


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


@pytest.mark.asyncio
async def test_each_fact_can_keep_its_own_historical_date(graph):
    """A mixed import retains individual dates and a fallback for undated entries."""
    april = datetime(2026, 4, 10, tzinfo=timezone.utc)
    july = datetime(2026, 7, 3, tzinfo=timezone.utc)
    fallback = datetime(2026, 6, 1, tzinfo=timezone.utc)
    first = _fact(graph, "Max", "PLANNED", "a", "Max planned a in April")
    first.valid_at = april
    second = _fact(graph, "Max", "FINISHED", "b", "Max finished b in July")
    second.valid_at = july
    third = _fact(graph, "Max", "STARTED", "c", "Max started c in June")
    result = await graph.save_facts(facts=[first, second, third], valid_at=fallback)
    assert [datetime.fromisoformat(f["valid_at"]) for f in result["facts"]] == [
        april, july, fallback]
    assert all(datetime.fromisoformat(f["recorded_at"]) > july for f in result["facts"])
    inventory = await graph.list_facts()
    assert {f["fact_id"]: f["valid_at"] for f in inventory["facts"]} == {
        f["fact_id"]: f["valid_at"] for f in result["facts"]}


@pytest.mark.asyncio
async def test_explicit_unknown_date_does_not_become_the_import_date(graph):
    """Explicit null overrides the batch fallback and remains unknown on retrieval."""
    fact = _fact(graph, "Max", "PLANNED", "a", "Max planned a sometime that spring")
    fact.valid_at = None
    result = await graph.save_facts(
        facts=[fact], valid_at=datetime(2026, 5, 1, tzinfo=timezone.utc))
    assert result["facts"][0]["valid_at"] is None
    assert result["facts"][0]["recorded_at"]
    assert (await graph.search_facts("Max"))[0]["valid_at"] is None


@pytest.mark.asyncio
async def test_omitted_date_still_defaults_to_the_recording_time(graph, monkeypatch):
    """Existing callers retain the current-time default when they omit dates."""
    when = datetime(2026, 9, 12, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(graph.store, "now", lambda: when)
    result = await graph.save_facts(facts=[
        _fact(graph, "Max", "PREFERS", "a", "Max prefers a")])
    assert result["facts"][0]["valid_at"] == result["facts"][0]["recorded_at"] == when.isoformat()


@pytest.mark.asyncio
async def test_dates_without_a_timezone_are_interpreted_as_utc(graph):
    """Date-only imports receive consistent UTC timestamps on both date paths."""
    when = datetime(2026, 5, 1)
    first = _fact(graph, "Max", "PLANNED", "a", "Max planned a")
    first.valid_at = when
    second = _fact(graph, "Max", "PLANNED", "b", "Max planned b")
    result = await graph.save_facts(facts=[first, second], valid_at=when)
    assert {fact["valid_at"] for fact in result["facts"]} == {
        when.replace(tzinfo=timezone.utc).isoformat()}


@pytest.mark.asyncio
async def test_fact_dates_survive_mcp_tool_validation(graph):
    """The MCP boundary preserves historical, unknown, and omitted date inputs."""
    from fastmcp import Client

    base = {"subject": "Max", "relation": "PLANNED", "fact": "Max made a plan"}
    async with Client(graph.mcp) as client:
        result = await client.call_tool("save_facts", {
            "facts": [
                {**base, "object": "April", "valid_at": "2026-04-10T00:00:00Z"},
                {**base, "object": "unknown", "valid_at": None},
                {**base, "object": "fallback"},
            ],
            "valid_at": "2026-06-01T00:00:00Z",
        })
    facts = result.structured_content["facts"]
    assert [fact["valid_at"] for fact in facts] == [
        "2026-04-10T00:00:00+00:00", None, "2026-06-01T00:00:00+00:00"]


@pytest.mark.asyncio
async def test_date_correction_retains_the_fact_story_and_embedding(embedded):
    """Repairing an import preserves identity, recording time, search, and provenance."""
    text = "Max planned a in May"
    result = await embedded.save_facts(
        facts=[_fact(embedded, "Max", "PLANNED", "a", text)],
        story="The dated source document.", source="archive/2026-05-01.md")
    original = result["facts"][0]
    before = await embedded.EntityEdge.get_by_uuid(embedded.store.driver(), original["fact_id"])
    await before.load_fact_embedding(embedded.store.driver())
    assert before.fact_embedding
    when = datetime(2026, 5, 1, tzinfo=timezone.utc)
    fixed = await embedded.set_fact_valid_at(
        original["fact_id"], when, reason="The import used the recording date")
    assert fixed["fact_id"] == original["fact_id"]
    assert fixed["fact"] == text
    assert fixed["valid_at"] == when.isoformat()
    assert fixed["recorded_at"] == original["recorded_at"]
    assert fixed["episode_id"] == result["episode_id"]
    assert fixed["valid_at_corrections"][0]["previous_valid_at"] == original["valid_at"]
    assert fixed["valid_at_corrections"][0]["reason"] == "The import used the recording date"
    assert (await embedded.get_story(result["episode_id"]))["fact_count"] == 1
    assert (await embedded.search_facts(text))[0]["valid_at"] == when.isoformat()
    assert len((await embedded.list_facts())["facts"]) == 1
    repeated = await embedded.set_fact_valid_at(original["fact_id"], when)
    assert repeated["valid_at_corrections"] == fixed["valid_at_corrections"]
    unknown = await embedded.set_fact_valid_at(original["fact_id"], None, reason="Date uncertain")
    assert unknown["valid_at"] is None
    assert len(unknown["valid_at_corrections"]) == 2
    assert (await embedded.search_facts(text))[0]["valid_at"] is None
    restored = await embedded.set_fact_valid_at(original["fact_id"], when)
    assert restored["valid_at_corrections"][-1]["previous_valid_at"] is None
    after = await embedded.EntityEdge.get_by_uuid(embedded.store.driver(), original["fact_id"])
    await after.load_fact_embedding(embedded.store.driver())
    assert after.fact_embedding == before.fact_embedding


@pytest.mark.asyncio
async def test_date_correction_cannot_invert_a_facts_validity(graph):
    """A correction cannot move a start date beyond the date the fact stopped applying."""
    original_date = datetime(2026, 4, 1, tzinfo=timezone.utc)
    result = await graph.save_facts(
        facts=[_fact(graph, "Max", "PLANNED", "a", "Max planned a")],
        valid_at=original_date)
    fact_id = result["facts"][0]["fact_id"]
    await graph.supersede_fact(fact_id, invalid_at=datetime(2026, 5, 1, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="invalid_at"):
        await graph.set_fact_valid_at(fact_id, datetime(2026, 6, 1))
    stored = (await graph.list_facts())["facts"][0]
    assert stored["valid_at"] == original_date.isoformat()
    assert "valid_at_corrections" not in stored


@pytest.mark.asyncio
async def test_date_correction_cannot_modify_another_group(graph, monkeypatch):
    """A known ID is insufficient to modify a different memory group's date."""
    result = await graph.save_facts(facts=[
        _fact(graph, "Max", "PLANNED", "a", "Max planned a")])
    monkeypatch.setattr(graph.store, "GROUP_ID", "other")
    with pytest.raises(ValueError, match="group"):
        await graph.set_fact_valid_at(result["facts"][0]["fact_id"], None)


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
@pytest.mark.parametrize("mode", ["keyword", "semantic"])
async def test_superseded_facts_do_not_consume_the_search_limit(graph, monkeypatch, mode):
    """Each search mode must return a current match behind a higher-ranked old fact."""
    if mode == "semantic":
        async def embedding(text: str) -> list[float]:
            """Return a vector for text, ranking coffee above tea for the query."""
            return [0.8, 0.6] if "tea" in text else [1.0, 0.0]

        monkeypatch.setattr(graph.embed, "embed", embedding)

    old = await graph.save_facts(facts=[
        _fact(graph, "Heather", "PREFERS", "coffee", "Heather likes coffee")])
    old_id = old["facts"][0]["fact_id"]
    await graph.supersede_fact(old_id)
    current = await graph.save_facts(facts=[
        _fact(graph, "Heather", "PREFERS", "tea",
              "Heather likes tea with breakfast every morning before starting work")])
    current_id = current["facts"][0]["fact_id"]
    query = "Heather" if mode == "keyword" else "preferred beverage"

    history = await graph.search_facts(query, limit=1, include_superseded=True)
    assert [hit["fact_id"] for hit in history] == [old_id]
    hits = await graph.search_facts(query, limit=1)
    assert [hit["fact_id"] for hit in hits] == [current_id]
    assert hits[0]["superseded"] is False


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


@pytest.mark.asyncio
async def test_inventory_pages_through_all_facts_including_history(graph, monkeypatch):
    """An audit visits each fact once without a query or an embedding service."""
    edge_model = graph.store.EntityEdge
    ids = iter([
        "f655d1e4-03ce-4a4f-a9b7-9104a03bfb95",
        "f4678cfd-1682-4e4d-ab41-f887b80776fb",
        "4e17917b-4ae8-4360-8d14-d3a1ad91c50e",
        "13e2080d-0c6f-4f00-9bfe-7abc557b1af9",
        "031ded59-004f-4b69-b906-607810a493e2",
    ])

    def edge_with_id(**kwargs):
        """Build an edge with UUIDs that exercise shared string-index prefixes."""
        return edge_model(uuid=next(ids), **kwargs)

    monkeypatch.setattr(graph.store, "EntityEdge", edge_with_id)
    result = await graph.save_facts(facts=[
        _fact(graph, "Max", "RECORDED", f"item {i}", f"Max recorded item {i}")
        for i in range(5)])
    expected = sorted((fact["fact_id"] for fact in result["facts"]), reverse=True)
    await graph.supersede_fact(expected[0])

    async def forbidden_embedding(text):
        """Fail if an inventory attempts to embed any text."""
        raise AssertionError("An inventory must not use embeddings")

    monkeypatch.setattr(graph.embed, "embed", forbidden_embedding)
    first = await graph.list_facts(limit=2)
    second = await graph.list_facts(limit=2, cursor=first["next_cursor"])
    third = await graph.list_facts(limit=2, cursor=second["next_cursor"])
    pages = [first, second, third]
    assert [len(page["facts"]) for page in pages] == [2, 2, 1]
    assert [fact["fact_id"] for page in pages for fact in page["facts"]] == expected
    assert first["facts"][0]["superseded"] is True
    assert first["next_cursor"] == expected[1]
    assert second["next_cursor"] == expected[3]
    assert third["next_cursor"] is None
    assert await graph.list_facts(cursor=expected[-1]) == {
        "facts": [], "next_cursor": None}


@pytest.mark.asyncio
async def test_inventory_is_empty_for_an_empty_graph(graph):
    """An empty store produces a terminal empty page."""
    assert await graph.list_facts() == {"facts": [], "next_cursor": None}


@pytest.mark.asyncio
async def test_inventory_only_lists_the_configured_group(graph, monkeypatch):
    """An audit cannot enumerate facts belonging to another memory group."""
    await graph.save_facts(facts=[_fact(graph, "Max", "KNOWS", "a", "Max knows a")])
    monkeypatch.setattr(graph.store, "GROUP_ID", "other")
    assert await graph.list_facts() == {"facts": [], "next_cursor": None}


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1, 101])
async def test_inventory_rejects_unbounded_page_sizes(graph, limit):
    """Page limits are bounded even when calling the implementation directly."""
    with pytest.raises(ValueError, match="limit"):
        await graph.list_facts(limit=limit)


@pytest.mark.asyncio
async def test_inventory_rejects_invalid_cursors(graph):
    """Malformed cursors fail clearly instead of returning misleading pages."""
    with pytest.raises(ValueError):
        await graph.list_facts(cursor="not-a-cursor")
