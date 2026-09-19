"""Knowledge-graph memory: agent-authored facts with a timeline.

The premise of this service is that the *agent* decides what is stored, how it
is typed, and when it stopped being true — no extraction pass, no LLM in the
write path. These tests pin that: every fact here is constructed by hand, with
schema invented on the spot, and nothing configures a model.
"""

import asyncio
import math
import os
import uuid
from datetime import datetime, timedelta, timezone

import falkordb
import pytest
from redis.exceptions import RedisError, ResponseError

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
async def test_entity_type_updates_preserve_summary_and_merge_attributes(graph):
    """Type normalization retains entity identity and the existing metadata contract."""
    original = await graph.store.upsert_entity("Workshop", "Place", attributes={"original": "kept"})
    enriched = await graph.store.upsert_entity(
        "Workshop", "Chat room", summary="First description", attributes={"added": "new"})
    assert enriched.uuid == original.uuid
    unchanged = await graph.store.upsert_entity("Workshop", "Chat_room", summary="Replacement")
    assert unchanged.uuid == original.uuid
    assert unchanged.summary == "First description"
    assert unchanged.attributes == {"original": "kept", "added": "new"}
    assert set(unchanged.labels) == {"Entity", "Place", "Chat_room"}


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


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_field", ["subject_type", "object_type"])
@pytest.mark.parametrize("reuse_story", [False, True])
async def test_invalid_batch_labels_leave_no_partial_writes(graph, monkeypatch, invalid_field,
                                                            reuse_story):
    """A bad fifth label leaves no facts, entity changes, or provenance to duplicate on retry."""
    from unittest.mock import AsyncMock

    original = await graph.save_facts(
        facts=[_fact(graph, "Existing", "KEEPS", "Original", "Existing keeps Original")],
        story="Original story", source="original-source")

    async def snapshot():
        """Return all stored node labels and properties plus relationship properties."""
        nodes, _, _ = await graph.store.driver().execute_query(
            "MATCH (n) RETURN n.uuid AS uuid, labels(n) AS labels, properties(n) AS properties "
            "ORDER BY uuid")
        edges, _, _ = await graph.store.driver().execute_query(
            "MATCH ()-[e]->() RETURN e.uuid AS uuid, properties(e) AS properties ORDER BY uuid")
        return nodes, edges

    before = await snapshot()
    facts = [_fact(graph, "Existing", "KEEPS", f"New {i}", f"Existing keeps New {i}",
                   s_type="Additional type") for i in range(4)]
    last = _fact(graph, "New subject", "KNOWS", "New object", "New subject knows New object")
    setattr(last, invalid_field, "Invalid` label")
    facts.append(last)
    provenance = ({"episode_id": original["episode_id"]} if reuse_story
                  else {"story": "New batch story", "source": "batch-source"})
    with monkeypatch.context() as scoped:
        embedding = AsyncMock(return_value=None)
        initialization = AsyncMock()
        scoped.setattr(graph.embed, "embed", embedding)
        scoped.setattr(graph.store, "ensure_indices", initialization)
        with pytest.raises(ValueError, match="node_labels"):
            await graph.save_facts(facts=facts, **provenance)
        assert await snapshot() == before
        embedding.assert_not_awaited()
        initialization.assert_not_awaited()

    setattr(last, invalid_field, "Valid label")
    saved = await graph.save_facts(facts=facts, **provenance)
    inventory = (await graph.list_facts())["facts"]
    assert len(inventory) == 6
    assert len({fact["fact_id"] for fact in inventory}) == 6
    assert len(saved["facts"]) == 5
    story = await graph.get_story(saved["episode_id"])
    assert story["fact_count"] == (6 if reuse_story else 5)
    assert story["story"] == ("Original story" if reuse_story else "New batch story")


@pytest.mark.asyncio
async def test_invalid_batch_labels_are_rejected_over_mcp_before_any_write(graph):
    """Tool input validation rejects a late bad type without storing a prefix or story."""
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    async with Client(graph.mcp) as client:
        with pytest.raises(ToolError, match="node_labels"):
            await client.call_tool("save_facts", {"facts": [
                {"subject": "A", "relation": "KNOWS", "object": "B", "fact": "A knows B"},
                {"subject": "B", "relation": "KNOWS", "object": "C", "fact": "B knows C",
                 "object_type": "Invalid` label"},
            ], "story": "Rejected story"})
    records, _, _ = await graph.store.driver().execute_query("MATCH (n) RETURN count(n) AS count")
    assert records == [{"count": 0}]


@pytest.mark.asyncio
@pytest.mark.parametrize("provenance", [{}, {"story": "Unused", "source": "Unused"},
                                        {"episode_id": "unused-episode"}])
async def test_empty_batches_remain_noops(graph, monkeypatch, provenance):
    """An empty batch neither creates nor looks up provenance and needs no services."""
    from unittest.mock import AsyncMock, Mock

    driver = Mock(side_effect=AssertionError("Empty batches must not access the graph"))
    initialize = AsyncMock()
    embedding = AsyncMock()
    monkeypatch.setattr(graph.store, "driver", driver)
    monkeypatch.setattr(graph.store, "ensure_indices", initialize)
    monkeypatch.setattr(graph.embed, "embed", embedding)
    assert await graph.save_facts(facts=[], **provenance) == {
        "facts": [], "episode_id": provenance.get("episode_id")}
    driver.assert_not_called()
    initialize.assert_not_awaited()
    embedding.assert_not_awaited()


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

    async def fake(text, *, is_query=False):
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
        async def embedding(text: str, *, is_query: bool = False) -> list[float]:
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


@pytest.mark.asyncio
async def test_keyword_search_survives_a_vector_query_failure(embedded, monkeypatch):
    """A failing vector query must not prevent exact-name memory retrieval."""
    result = await embedded.save_facts(facts=[
        _fact(embedded, "Heather", "RUNS_ON", "model", "Heather runs on model")])

    async def unavailable(*args, **kwargs):
        """Simulate an unavailable vector index while the graph remains readable."""
        raise RuntimeError("Vector query unavailable")

    monkeypatch.setattr(embedded, "edge_similarity_search", unavailable)
    hits = await embedded.search_facts("Heather")
    assert [hit["fact_id"] for hit in hits] == [result["facts"][0]["fact_id"]]


@pytest.mark.asyncio
async def test_semantic_threshold_can_be_tuned_for_the_embedding_model(graph, monkeypatch):
    """Model-specific score thresholds control semantic recall without keyword matches."""
    async def vector(text, *, is_query=False):
        """Return query/document vectors whose cosine similarity is one half."""
        return [1.0, 0.0] if is_query else [0.5, math.sqrt(0.75)]

    monkeypatch.setattr(graph.embed, "embed", vector)
    result = await graph.save_facts(facts=[
        _fact(graph, "Heather", "PREFERS", "tea", "Heather prefers tea")])
    monkeypatch.setattr(graph.embed, "MIN_SCORE", 0.8)
    assert await graph.search_facts("favored beverage") == []
    monkeypatch.setattr(graph.embed, "MIN_SCORE", 0.7)
    hits = await graph.search_facts("favored beverage")
    assert [hit["fact_id"] for hit in hits] == [result["facts"][0]["fact_id"]]


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


async def _save_links(graph, links):
    """Store synthetic directed relationships and return their save response."""
    return await graph.save_facts(facts=[
        graph.Fact(subject=source, relation=relation, object=target,
                   fact=f"{source} {relation} {target}")
        for source, relation, target in links])


@pytest.mark.asyncio
async def test_neighborhood_follows_multiple_hops_and_preserves_directions(graph):
    """A bounded neighborhood includes incoming links without reversing their meaning."""
    await _save_links(graph, [("A", "USES", "B"), ("C", "OWNS", "B"),
                              ("B", "NEEDS", "D"), ("D", "NEEDS", "E")])
    result = await graph.get_neighborhood("A", max_hops=2)
    assert result["found"] and not result["truncated"]
    assert {node["name"]: node["hops"] for node in result["nodes"]} == {
        "A": 0, "B": 1, "C": 2, "D": 2}
    assert {(fact["subject"], fact["object"]) for fact in result["facts"]} == {
        ("A", "B"), ("C", "B"), ("B", "D")}
    ids = {node["entity_id"] for node in result["nodes"]}
    assert all(fact["subject_id"] in ids and fact["object_id"] in ids for fact in result["facts"])


@pytest.mark.asyncio
@pytest.mark.parametrize("direction,expected", [
    ("outgoing", {"B", "D"}), ("incoming", {"B", "A", "C"}),
])
async def test_neighborhood_direction_controls_expansion(graph, direction, expected):
    """Incoming and outgoing exploration follow different recorded relationships."""
    await _save_links(graph, [("A", "USES", "B"), ("C", "OWNS", "B"), ("B", "NEEDS", "D")])
    result = await graph.get_neighborhood("B", max_hops=1, direction=direction)
    assert {node["name"] for node in result["nodes"]} == expected


@pytest.mark.asyncio
async def test_traversal_survives_relationship_group_index_misses(graph, monkeypatch):
    """Indexed group misses must not hide a root or an intermediate entity's facts."""
    from types import SimpleNamespace

    saved = await _save_links(graph, [("Max", "FRIEND_OF", "Nanhi"),
                                     ("Heather", "KNOWS", "Max"),
                                     ("Max", "OLD", "Former plan")])
    await graph.supersede_fact(saved["facts"][2]["fact_id"])
    root = await graph.store.find_entity("Max")
    graph_type = type(graph.store.driver().client.select_graph(graph.store.GRAPH_DATABASE))
    read = graph_type.ro_query

    async def missing_group_index(self, query, params=None, **kwargs):
        """Simulate the live indexed-group miss while preserving real graph reads."""
        if (root.uuid in (params or {}).get("frontier", [])
                and "e.group_id = $group_id" in query):
            return SimpleNamespace(header=[], result_set=[])
        return await read(self, query, params=params, **kwargs)

    monkeypatch.setattr(graph_type, "ro_query", missing_group_index)
    assert len((await graph.get_entity("Max"))["facts"]) == 3
    neighborhood = await graph.get_neighborhood("Max", max_hops=1)
    assert {fact["fact_id"] for fact in neighborhood["facts"]} == {
        fact["fact_id"] for fact in saved["facts"][:2]}
    assert not neighborhood["truncated"]
    for direction, other in [("outgoing", "Nanhi"), ("incoming", "Heather")]:
        result = await graph.get_neighborhood("Max", max_hops=1, direction=direction)
        assert {node["name"] for node in result["nodes"]} == {"Max", other}
    historical = await graph.get_neighborhood("Max", max_hops=1, include_superseded=True)
    assert len(historical["facts"]) == 3
    for source, target, hops in [("Max", "Nanhi", 1), ("Heather", "Nanhi", 2)]:
        path = await graph.find_path(source, target, directed=True)
        assert path["found"] and path["hops"] == hops
        assert all(fact["traversed_forward"] for fact in path["facts"])
    assert (await graph.find_path("Nanhi", "Heather"))["hops"] == 2


@pytest.mark.asyncio
async def test_exploration_handles_cycles_self_links_and_parallel_facts(graph, monkeypatch):
    """Cycles do not duplicate facts or trigger embedding calls during graph reads."""
    saved = await _save_links(graph, [("A", "LIKES", "B"), ("A", "KNOWS", "B"),
                                     ("B", "KNOWS", "C"), ("C", "KNOWS", "A"),
                                     ("B", "REFLECTS_ON", "B")])

    async def forbidden(*args, **kwargs):
        """Fail if graph-only exploration attempts to call an embedding model."""
        raise AssertionError("Graph exploration must not embed text")

    monkeypatch.setattr(graph.embed, "embed", forbidden)
    neighborhood = await graph.get_neighborhood("A", max_hops=6)
    assert len(neighborhood["nodes"]) == 3
    assert {fact["fact_id"] for fact in neighborhood["facts"]} == {
        fact["fact_id"] for fact in saved["facts"]}
    assert not neighborhood["truncated"]
    path = await graph.find_path("A", "B")
    assert path["hops"] == 1 and len(path["facts"]) == 1
    assert (await graph.explain_fact(path["facts"][0]["fact_id"]))["found"]
    assert len((await graph.list_facts())["facts"]) == 5


@pytest.mark.asyncio
async def test_superseded_links_cannot_bridge_a_current_exploration(graph):
    """Filtering happens before traversal, including when the current route is longer."""
    saved = await _save_links(graph, [("A", "OLD", "B"), ("B", "NEXT", "C"),
                                     ("A", "NEW", "D"), ("D", "NEXT", "E"),
                                     ("E", "NEXT", "C")])
    await graph.supersede_fact(saved["facts"][0]["fact_id"])
    neighborhood = await graph.get_neighborhood("A", max_hops=2)
    assert {node["name"] for node in neighborhood["nodes"]} == {"A", "D", "E"}
    current = await graph.find_path("A", "C", max_hops=3)
    assert [node["name"] for node in current["nodes"]] == ["A", "D", "E", "C"]
    assert all(not fact["superseded"] for fact in current["facts"])
    historical = await graph.find_path("A", "C", include_superseded=True)
    assert [node["name"] for node in historical["nodes"]] == ["A", "B", "C"]
    assert historical["facts"][0]["superseded"]


@pytest.mark.asyncio
async def test_path_reports_reverse_traversal_without_reversing_the_fact(graph):
    """An undirected connection is not a newly asserted reverse relationship."""
    await _save_links(graph, [("A", "OWNS", "B")])
    reverse = await graph.find_path("B", "A")
    assert reverse["found"] and reverse["hops"] == 1
    fact = reverse["facts"][0]
    assert (fact["subject"], fact["relation"], fact["object"]) == ("A", "OWNS", "B")
    assert fact["traversed_forward"] is False
    assert fact["from_entity_id"] == fact["object_id"]
    assert fact["to_entity_id"] == fact["subject_id"]
    assert not (await graph.find_path("B", "A", directed=True))["found"]
    assert (await graph.find_path("A", "B", directed=True))["facts"][0]["traversed_forward"]


@pytest.mark.asyncio
async def test_path_prefers_the_shortest_eligible_route(graph):
    """The result uses the shorter of two eligible routes and respects the hop cap."""
    await _save_links(graph, [("A", "NEXT", "B"), ("B", "NEXT", "D"),
                              ("A", "NEXT", "C"), ("C", "NEXT", "E"), ("E", "NEXT", "D")])
    result = await graph.find_path("A", "D", directed=True)
    assert [node["name"] for node in result["nodes"]] == ["A", "B", "D"]
    assert result["hops"] == 2
    too_short = await graph.find_path("A", "D", max_hops=1)
    assert not too_short["found"] and too_short["max_hops"] == 1


@pytest.mark.asyncio
async def test_fact_budget_reports_incomplete_search_and_exact_completion(graph):
    """The budget flag distinguishes an unexplored frontier from a complete small graph."""
    await _save_links(graph, [("A", "NEXT", "B"), ("B", "NEXT", "C")])
    limited = await graph.get_neighborhood("A", max_hops=2, limit=1)
    assert len(limited["facts"]) == 1 and limited["truncated"]
    radius_one = await graph.get_neighborhood("A", max_hops=1, limit=1)
    assert not radius_one["truncated"]
    complete = await graph.get_neighborhood("A", max_hops=3, limit=2)
    assert len(complete["facts"]) == 2 and not complete["truncated"]
    path = await graph.find_path("A", "C", limit=1)
    assert not path["found"] and path["truncated"] and path["explored_facts"] == 1


@pytest.mark.asyncio
async def test_direct_target_is_prioritized_with_a_small_fact_budget(graph):
    """A direct target remains discoverable even when other adjacent facts exceed the budget."""
    await _save_links(graph, [("A", "KNOWS", "B"), ("A", "KNOWS", "C"), ("A", "KNOWS", "D")])
    path = await graph.find_path("A", "D", limit=1)
    assert path["found"] and path["hops"] == 1
    assert path["explored_facts"] == 1 and path["truncated"]


@pytest.mark.asyncio
async def test_exploration_handles_missing_isolated_and_identical_entities(graph):
    """Missing entities are explicit, while a known entity has a zero-hop path to itself."""
    assert not (await graph.get_neighborhood("missing"))["found"]
    missing = await graph.find_path("missing", "also missing")
    assert missing["missing_entities"] == ["missing", "also missing"]
    await graph.store.upsert_entity("A", "Thing")
    isolated = await graph.get_neighborhood("A")
    assert isolated["found"] and isolated["facts"] == [] and not isolated["truncated"]
    assert isolated["nodes"][0]["hops"] == 0
    same = await graph.find_path("A", "A")
    assert same["found"] and same["hops"] == 0 and same["facts"] == []


@pytest.mark.asyncio
async def test_exploration_cannot_cross_group_boundaries(graph, monkeypatch):
    """Even a malformed cross-group edge cannot expose foreign entities or provenance."""
    await _save_links(graph, [("A", "KNOWS", "B")])
    local = await graph.store.find_entity("A")
    with monkeypatch.context() as scoped:
        scoped.setattr(graph.store, "GROUP_ID", "other")
        foreign_facts = await _save_links(graph, [("X", "KNOWS", "Y")])
        foreign = await graph.store.find_entity("X")
    cross_group = await graph.store.save_edge(local, "CROSS_GROUP", foreign, "A knows X")
    result = await graph.get_neighborhood("A", max_hops=6)
    assert {node["name"] for node in result["nodes"]} == {"A", "B"}
    assert (await graph.find_path("A", "X"))["missing_entities"] == ["X"]
    with pytest.raises(ValueError, match="group"):
        await graph.explain_fact(foreign_facts["facts"][0]["fact_id"])
    with pytest.raises(RuntimeError):
        await graph.explain_fact(cross_group.uuid)


@pytest.mark.asyncio
@pytest.mark.parametrize("method,arguments", [
    ("get_neighborhood", {"name": "A", "max_hops": 0}),
    ("get_neighborhood", {"name": "A", "max_hops": 7}),
    ("get_neighborhood", {"name": "A", "limit": 0}),
    ("get_neighborhood", {"name": "A", "limit": 101}),
    ("get_neighborhood", {"name": "A", "direction": "sideways"}),
    ("find_path", {"source": "A", "target": "B", "max_hops": 7}),
    ("find_path", {"source": "A", "target": "B", "limit": 101}),
])
async def test_exploration_rejects_unbounded_settings(graph, method, arguments):
    """Invalid budgets are rejected even for direct Python calls."""
    with pytest.raises(ValueError):
        await getattr(graph, method)(**arguments)


@pytest.mark.asyncio
async def test_explanation_preserves_evidence_provenance_and_date_history(graph):
    """Explanations expose stored reasoning and history without synthesizing evidence."""
    saved = await graph.save_facts(facts=[graph.Fact(
        subject="A", relation="PLANS", object="B", fact="A appears to be planning B",
        evidence_kind="inferred", rationale="A requested components used by B")],
        story="The original conversation supplied the context.", source="messages/123")
    fact_id = saved["facts"][0]["fact_id"]
    await graph.set_fact_valid_at(fact_id, datetime(2026, 5, 1, tzinfo=timezone.utc), reason="Source date")
    await graph.supersede_fact(fact_id, reason="The plan was cancelled")
    result = await graph.explain_fact(fact_id)
    assert result["found"]
    assert result["fact"]["evidence_kind"] == "inferred"
    assert result["fact"]["rationale"] == "A requested components used by B"
    assert result["fact"]["valid_at_corrections"][0]["reason"] == "Source date"
    assert result["superseded_reason"] == "The plan was cancelled"
    assert result["sources"][0]["source"] == "messages/123"
    assert result["sources"][0]["story"] == "The original conversation supplied the context."
    assert result["missing_episode_ids"] == [] and not result["sources_truncated"]


@pytest.mark.asyncio
async def test_explanation_bounds_stories_and_preserves_source_only_attribution(graph):
    """Narrative and attribution truncation are explicit and full stories remain readable."""
    saved = await graph.save_facts(facts=[_fact(graph, "A", "KNOWS", "B", "A knows B")],
                                   story="Long narrative", source="s" * 1001)
    result = await graph.explain_fact(saved["facts"][0]["fact_id"], story_chars=4)
    source = result["sources"][0]
    assert source["story"] == "Long" and source["story_truncated"]
    assert len(source["source"]) == 1000 and source["source_truncated"]
    assert (await graph.get_story(source["episode_id"]))["story"] == "Long narrative"
    source_only = await graph.save_facts(
        facts=[_fact(graph, "A", "KNOWS", "C", "A knows C")], source="messages/456")
    provenance = (await graph.explain_fact(source_only["facts"][0]["fact_id"]))["sources"][0]
    assert provenance["story"] == "" and not provenance["story_truncated"]
    assert provenance["source"] == "messages/456"


@pytest.mark.asyncio
async def test_explanation_reports_missing_sources_and_limits_episode_reads(graph, monkeypatch):
    """Missing or foreign provenance is not fabricated, and source limits remain bounded."""
    saved = await _save_links(graph, [("A", "KNOWS", "B")])
    first = await graph.store.save_episode("First story", "first")
    second = await graph.store.save_episode("Second story", "second")
    with monkeypatch.context() as scoped:
        scoped.setattr(graph.store, "GROUP_ID", "other")
        foreign = await graph.store.save_episode("Private story", "foreign")
    edge = await graph.EntityEdge.get_by_uuid(graph.store.driver(), saved["facts"][0]["fact_id"])
    missing_id = str(uuid.uuid4())
    edge.episodes = [first.uuid, first.uuid, missing_id, second.uuid, foreign.uuid]
    await graph.store.update_edge(edge)
    limited = await graph.explain_fact(edge.uuid, source_limit=2, story_chars=0)
    assert len(limited["sources"]) == 1 and limited["sources_truncated"]
    assert limited["missing_episode_ids"] == [missing_id]
    assert limited["sources"][0]["story_truncated"] and limited["sources"][0]["story"] == ""
    complete = await graph.explain_fact(edge.uuid, source_limit=10)
    assert len(complete["sources"]) == 2 and not complete["sources_truncated"]
    assert complete["missing_episode_ids"] == [missing_id, foreign.uuid]


@pytest.mark.asyncio
async def test_explanation_does_not_assign_evidence_to_legacy_facts(graph):
    """Legacy records with no evidence metadata remain explicitly unspecified."""
    saved = await _save_links(graph, [("A", "KNOWS", "B")])
    edge = await graph.EntityEdge.get_by_uuid(graph.store.driver(), saved["facts"][0]["fact_id"])
    edge.attributes = {}
    await graph.store.update_edge(edge)
    result = await graph.explain_fact(edge.uuid)
    assert result["fact"]["evidence_kind"] == "unspecified"
    assert result["fact"]["rationale"] == "" and result["sources"] == []
    assert not (await graph.explain_fact(str(uuid.uuid4())))["found"]


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [
    {"source_limit": 0}, {"source_limit": 11}, {"story_chars": -1}, {"story_chars": 5001},
])
async def test_explanation_rejects_unbounded_settings(graph, arguments):
    """Explanation limits are validated before loading a fact."""
    with pytest.raises(ValueError):
        await graph.explain_fact(str(uuid.uuid4()), **arguments)


@pytest.mark.asyncio
@pytest.mark.parametrize("method,arguments", [
    ("get_neighborhood", {"name": "A"}),
    ("find_path", {"source": "A", "target": "B"}),
])
async def test_exploration_timeout_is_not_reported_as_absence(graph, monkeypatch, method, arguments):
    """A timed-out search raises instead of claiming the graph has no connection."""
    async def slow_lookup(name):
        """Simulate an unresponsive graph lookup."""
        await asyncio.sleep(1)

    monkeypatch.setattr(graph.store, "find_entity", slow_lookup)
    monkeypatch.setattr(graph.traversal, "TIMEOUT_SECONDS", 0.01)
    with pytest.raises(TimeoutError):
        await getattr(graph, method)(**arguments)


@pytest.mark.asyncio
async def test_new_graph_tools_are_available_over_mcp(graph):
    """MCP exposes read-only graph tools with input bounds and persisted evidence metadata."""
    from fastmcp import Client

    async with Client(graph.mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        for name in ["get_neighborhood", "find_path", "explain_fact"]:
            assert tools[name].annotations.readOnlyHint is True
        assert tools["get_neighborhood"].inputSchema["properties"]["max_hops"]["maximum"] == 6
        saved = await client.call_tool("save_facts", {"facts": [{
            "subject": "A", "relation": "KNOWS", "object": "B", "fact": "A knows B",
            "evidence_kind": "reported", "rationale": "A said so",
        }]})
        neighborhood = await client.call_tool("get_neighborhood", {"name": "A"})
        assert neighborhood.structured_content["facts"][0]["evidence_kind"] == "reported"
        path = await client.call_tool("find_path", {"source": "A", "target": "B"})
        assert path.structured_content["hops"] == 1
        explained = await client.call_tool("explain_fact", {
            "fact_id": saved.structured_content["facts"][0]["fact_id"],
        })
        assert explained.structured_content["fact"]["rationale"] == "A said so"


@pytest.mark.asyncio
async def test_traversal_uses_a_database_enforced_read_timeout(graph, monkeypatch):
    """Each frontier is a read-only query with a database-side execution deadline."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from graphiti_core.driver.driver import GraphProvider

    query = AsyncMock(return_value=SimpleNamespace(header=[], result_set=[]))
    client = SimpleNamespace(select_graph=Mock(return_value=SimpleNamespace(ro_query=query)))
    driver = SimpleNamespace(client=client, provider=GraphProvider.FALKORDB)
    monkeypatch.setattr(graph.store, "driver", lambda: driver)
    assert await graph.store.adjacent_edges(["entity-id"], [], 7) == []
    assert query.await_args.kwargs["timeout"] == 2000
    assert query.await_args.kwargs["params"]["limit"] == 7
    client.select_graph.assert_called_once_with(graph.store.GRAPH_DATABASE)
    assert await graph.store.adjacent_edges([], [], 7) == []
    assert query.await_count == 1
    with pytest.raises(ValueError, match="direction"):
        await graph.store.adjacent_edges(["entity-id"], [], 7, direction="invalid")


@pytest.mark.asyncio
@pytest.mark.parametrize("message,expected", [("Query timed out", TimeoutError),
                                            ("Syntax error", ResponseError)])
async def test_database_errors_do_not_become_empty_traversals(graph, monkeypatch, message, expected):
    """Database timeouts are explicit and unrelated failures are not swallowed."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from graphiti_core.driver.driver import GraphProvider

    error = ResponseError(message)
    query = AsyncMock(side_effect=error)
    driver = SimpleNamespace(provider=GraphProvider.FALKORDB,
        client=SimpleNamespace(select_graph=Mock(return_value=SimpleNamespace(ro_query=query))))
    monkeypatch.setattr(graph.store, "driver", lambda: driver)
    with pytest.raises(expected):
        await graph.store.adjacent_edges(["entity-id"], [], 7)


@pytest.mark.asyncio
async def test_explanation_respects_the_overall_timeout(graph, monkeypatch):
    """An unavailable fact read cannot hold the explanation call indefinitely."""
    async def slow_read(*args, **kwargs):
        """Simulate an unresponsive graph read."""
        await asyncio.sleep(1)

    monkeypatch.setattr(graph.store, "get_edge", slow_read)
    monkeypatch.setattr(graph.traversal, "TIMEOUT_SECONDS", 0.01)
    with pytest.raises(TimeoutError):
        await graph.explain_fact(str(uuid.uuid4()))


@pytest.mark.asyncio
async def test_exploration_does_not_return_a_broken_node_snapshot(graph, monkeypatch):
    """Disappearing entities cause a retryable failure, not a malformed explanation graph."""
    await _save_links(graph, [("A", "KNOWS", "B")])

    async def missing_nodes(*args, **kwargs):
        """Simulate entities disappearing between frontier and result reads."""
        return []

    monkeypatch.setattr(graph.EntityNode, "get_by_uuids", missing_nodes)
    with pytest.raises(RuntimeError, match="Graph changed"):
        await graph.get_neighborhood("A")


def test_evidence_kind_rejects_unrecognized_certainty_labels():
    """Evidence categories cannot silently become arbitrary confidence claims."""
    from graphiti_mcp.server import Fact

    with pytest.raises(ValueError):
        Fact(subject="A", relation="KNOWS", object="B", fact="A knows B", evidence_kind="verified")


@pytest.mark.asyncio
async def test_three_todos_remain_mutable_when_indexed_uuid_reads_miss_them(graph, monkeypatch):
    """Search-visible facts can be corrected without indexed UUID reads or edge upserts."""
    from graphiti_core.errors import EdgeNotFoundError

    identifiers = [
        "e1c6f27d-0000-4000-8000-000000000001",
        "f90f21a5-0000-4000-8000-000000000002",
        "20f6386f-0000-4000-8000-000000000003",
    ]
    remaining_ids = iter(identifiers)
    edge_model = graph.store.EntityEdge

    def next_edge(**kwargs):
        """Use synthetic UUIDs with the prefixes from the reported batch."""
        return edge_model(uuid=next(remaining_ids), **kwargs)

    async def embedding(text, *, is_query=False):
        """Attach a known vector so metadata updates must preserve it."""
        return [0.6, 0.8, 0.0, 0.0]

    monkeypatch.setattr(graph.store, "EntityEdge", next_edge)
    monkeypatch.setattr(graph.embed, "embed", embedding)
    saved = await graph.save_facts(facts=[
        _fact(graph, "Max", "TODO", f"task {i}", f"Max has task {i} to do",
              o_type="Life task") for i in range(3)])
    execute = graph.store.driver().execute_query

    async def snapshot():
        """Read physical relationship identities and vectors without the UUID index."""
        records, _, _ = await execute(
            "MATCH (s:Entity)-[e:RELATES_TO]->(t:Entity) "
            "WHERE toString(e.uuid) IN $ids "
            "RETURN e.uuid AS uuid, id(e) AS relationship_id, s.uuid AS source_uuid, "
            "t.uuid AS target_uuid, e.fact AS fact, e.fact_embedding AS embedding, "
            "e.created_at AS created_at, e.valid_at AS valid_at, e.episodes AS episodes",
            ids=identifiers)
        return records

    before = {record["uuid"]: record for record in await snapshot()}
    assert len(before) == 3

    async def missing_index_entries(query, **params):
        """Reproduce the observed point-index miss while retaining real database reads."""
        if (params.get("uuid") in identifiers and "RELATES_TO" in query
                and "toString(e.uuid)" not in query):
            return [], [], None
        if "MERGE" in query:
            raise AssertionError("Updating a known fact must never upsert a relationship")
        return await execute(query, **params)

    monkeypatch.setattr(graph.store.driver(), "execute_query", missing_index_entries)
    for identifier in identifiers:
        with pytest.raises(EdgeNotFoundError):
            await edge_model.get_by_uuid(graph.store.driver(), identifier)
    assert {hit["fact_id"] for hit in await graph.search_facts("Max")} == set(identifiers)

    when = datetime(2026, 5, 1, tzinfo=timezone.utc)
    for original in saved["facts"]:
        fixed = await graph.set_fact_valid_at(original["fact_id"], when, reason="Source date")
        assert fixed["valid_at"] == when.isoformat()
        assert fixed["recorded_at"] == original["recorded_at"]
        assert fixed["fact"] == original["fact"]
        assert (await graph.explain_fact(original["fact_id"]))["found"]
    assert (await graph.supersede_fact(identifiers[0], reason="Completed"))["superseded"]

    after = await snapshot()
    assert len(after) == 3
    for record in after:
        assert record["valid_at"] == when.isoformat()
        for field in ["relationship_id", "source_uuid", "target_uuid", "fact",
                      "embedding", "created_at", "episodes"]:
            assert record[field] == before[record["uuid"]][field]


@pytest.mark.asyncio
async def test_scalar_fact_lookup_rejects_missing_partial_and_foreign_ids(graph, monkeypatch):
    """Exact ID lookup must not match prefixes or cross a memory group boundary."""
    from graphiti_core.errors import EdgeNotFoundError

    saved = await _save_links(graph, [("A", "KNOWS", "B")])
    identifier = saved["facts"][0]["fact_id"]
    assert (await graph.store.get_edge(identifier)).uuid == identifier
    for missing in [str(uuid.uuid4()), identifier[:8]]:
        with pytest.raises(EdgeNotFoundError):
            await graph.store.get_edge(missing)
    monkeypatch.setattr(graph.store, "GROUP_ID", "other")
    with pytest.raises(ValueError, match="group"):
        await graph.store.get_edge(identifier)


@pytest.mark.asyncio
async def test_metadata_updates_refuse_duplicate_uuid_records(graph):
    """A genuinely ambiguous ID cannot be read or mutate multiple relationships."""
    saved = await _save_links(graph, [("A", "KNOWS", "B")])
    identifier = saved["facts"][0]["fact_id"]
    edge = await graph.store.get_edge(identifier)
    original_date = edge.valid_at
    await graph.store.driver().execute_query(
        "MATCH (s:Entity)-[e:RELATES_TO]->(t:Entity) WHERE toString(e.uuid) = $uuid "
        "CREATE (s)-[duplicate:RELATES_TO]->(t) SET duplicate = properties(e)", uuid=identifier)
    with pytest.raises(ValueError, match="Multiple facts"):
        await graph.store.get_edge(identifier)
    edge.valid_at = datetime(2026, 5, 1, tzinfo=timezone.utc)
    with pytest.raises(RuntimeError, match="matching identity"):
        await graph.store.update_edge(edge)
    records, _, _ = await graph.store.driver().execute_query(
        "MATCH ()-[e:RELATES_TO]->() WHERE toString(e.uuid) = $uuid "
        "RETURN e.valid_at AS valid_at", uuid=identifier)
    assert len(records) == 2
    assert all(datetime.fromisoformat(record["valid_at"]) == original_date for record in records)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["missing", "endpoints", "group"])
async def test_metadata_updates_never_create_a_missing_or_reidentified_edge(graph, change):
    """Updating a stale model cannot mint a new edge or change its endpoints."""
    saved = await _save_links(graph, [("A", "KNOWS", "B")])
    edge = await graph.store.get_edge(saved["facts"][0]["fact_id"])
    if change == "missing":
        edge.uuid = str(uuid.uuid4())
    elif change == "endpoints":
        edge.target_node_uuid = edge.source_node_uuid
    else:
        edge.group_id = "other"
    with pytest.raises(ValueError if change == "group" else RuntimeError):
        await graph.store.update_edge(edge)
    facts = (await graph.list_facts())["facts"]
    assert len(facts) == 1
    assert facts[0]["fact_id"] == saved["facts"][0]["fact_id"]
    assert (facts[0]["subject"], facts[0]["object"]) == ("A", "B")
