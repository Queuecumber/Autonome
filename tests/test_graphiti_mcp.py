"""Knowledge-graph memory: agent-authored relationships with a timeline.

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


def _relationship(server, subject, relation, obj, text, s_type="Person", o_type="Topic"):
    return server.Relationship(subject=subject, subject_type=s_type, relation=relation,
                       object=obj, object_type=o_type, description=text)


# ── Writing ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_fact_keeps_its_sentence(graph):
    """The triple is an index over the sentence, not a replacement for it —
    nothing the agent wrote should be reduced to a schema."""
    r = await graph.save_memory(relationships=[_relationship(
        graph, "Max", "PREFERS", "commit granularity",
        "Max prefers function-level commit granularity")])
    stored = r["relationships"][0]
    assert stored["description"] == "Max prefers function-level commit granularity"
    assert (stored["subject"], stored["relation"], stored["object"]) == (
        "Max", "PREFERS", "commit granularity")


@pytest.mark.asyncio
async def test_schema_is_invented_not_declared(graph):
    """Entity types and relation names are free-form: a vocabulary the agent
    makes up mid-conversation costs no migration."""
    await graph.save_memory(relationships=[
        _relationship(graph, "Nanhi", "RUNS_ON", "kimi-k3", "Nanhi runs on kimi-k3",
              s_type="Agent", o_type="Model"),
        _relationship(graph, "Max", "COMMITTED_TO", "design review",
              "Max will review the design", o_type="Commitment"),
    ])
    vocab = await graph.list_vocabulary()
    assert {"Agent", "Model", "Commitment"} <= set(vocab["entity_types"])
    assert {"RUNS_ON", "COMMITTED_TO"} <= set(vocab["relations"])


@pytest.mark.asyncio
async def test_the_same_entity_is_reused_not_duplicated(graph):
    """Two relationships about Max must attach to one Max, or the graph fragments and
    `get_entity` only ever sees half of what is known."""
    await graph.save_memory(relationships=[_relationship(graph, "Max", "PREFERS", "a", "Max prefers a")])
    await graph.save_memory(relationships=[_relationship(graph, "Max", "DISLIKES", "b", "Max dislikes b")])
    entity = await graph.get_entity("Max")
    assert entity["found"] and len(entity["relationships"]) == 2


@pytest.mark.asyncio
async def test_an_existing_entity_gains_types_without_losing_them(graph):
    await graph.save_memory(relationships=[_relationship(graph, "Max", "IS", "engineer",
                                        "Max is an engineer", s_type="Person")])
    await graph.save_memory(relationships=[_relationship(graph, "Max", "REVIEWS", "PRs",
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
    result = await graph.save_memory(relationships=[
        _relationship(graph, "Workshop chat", "CONTAINS", "Copper sigil",
              "Workshop chat contains the Copper sigil", subject_type, object_type)])

    assert result["relationships"][0]["subject"] == "Workshop chat"
    assert result["relationships"][0]["object"] == "Copper sigil"
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
        await graph.save_memory(relationships=[
            _relationship(graph, "Workshop chat", "CONTAINS", "Copper sigil",
                  "Workshop chat contains the Copper sigil", subject_type, object_type)])

    room = await graph.get_entity("Workshop chat")
    artifact = await graph.get_entity("Copper sigil")
    assert sorted(room["types"]) == ["Chat_room", "Place"]
    assert artifact["types"] == ["Character_artifact"]
    assert len(room["relationships"]) == len(artifact["relationships"]) == 3
    assert (await graph.list_vocabulary())["entity_types"] == [
        "Character_artifact", "Chat_room", "Place"]


@pytest.mark.asyncio
async def test_whitespace_only_entity_types_are_omitted(graph):
    """Whitespace-only types behave like the optional empty type."""
    await graph.save_memory(relationships=[
        _relationship(graph, "Workshop chat", "CONTAINS", "Copper sigil",
              "Workshop chat contains the Copper sigil", " \t ", "\n")])

    assert (await graph.get_entity("Workshop chat"))["types"] == []
    assert (await graph.get_entity("Copper sigil"))["types"] == []
    assert (await graph.list_vocabulary())["entity_types"] == []


@pytest.mark.asyncio
async def test_entity_types_still_reject_punctuation(graph):
    """Whitespace normalization must retain validation of other label characters."""
    with pytest.raises(ValueError, match="node_labels"):
        await graph.save_memory(relationships=[
            _relationship(graph, "Workshop chat", "CONTAINS", "Copper sigil",
                  "Workshop chat contains the Copper sigil", "Chat` room", "Artifact")])


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_field", ["subject_type", "object_type"])
@pytest.mark.parametrize("reuse_story", [False, True])
async def test_invalid_batch_labels_leave_no_partial_writes(graph, monkeypatch, invalid_field,
                                                            reuse_story):
    """A bad fifth label leaves no relationships, entity changes, or provenance to duplicate on retry."""
    from unittest.mock import AsyncMock

    original = await graph.save_memory(
        relationships=[_relationship(graph, "Existing", "KEEPS", "Original", "Existing keeps Original")],
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
    relationships = [_relationship(graph, "Existing", "KEEPS", f"New {i}", f"Existing keeps New {i}",
                   s_type="Additional type") for i in range(4)]
    last = _relationship(graph, "New subject", "KNOWS", "New object", "New subject knows New object")
    setattr(last, invalid_field, "Invalid` label")
    relationships.append(last)
    provenance = ({"story_id": original["story_id"]} if reuse_story
                  else {"story": "New batch story", "source": "batch-source"})
    with monkeypatch.context() as scoped:
        embedding = AsyncMock(return_value=None)
        initialization = AsyncMock()
        scoped.setattr(graph.embed, "embed", embedding)
        scoped.setattr(graph.store, "ensure_indices", initialization)
        with pytest.raises(ValueError, match="node_labels"):
            await graph.save_memory(relationships=relationships, **provenance)
        assert await snapshot() == before
        embedding.assert_not_awaited()
        initialization.assert_not_awaited()

    setattr(last, invalid_field, "Valid label")
    saved = await graph.save_memory(relationships=relationships, **provenance)
    inventory = (await graph.list_relationships())["relationships"]
    assert len(inventory) == 6
    assert len({fact["relationship_id"] for fact in inventory}) == 6
    assert len(saved["relationships"]) == 5
    story = await graph.get_story(saved["story_id"])
    assert story["relationship_count"] == (6 if reuse_story else 5)
    assert story["story"] == ("Original story" if reuse_story else "New batch story")


@pytest.mark.asyncio
async def test_invalid_batch_labels_are_rejected_over_mcp_before_any_write(graph):
    """Tool input validation rejects a late bad type without storing a prefix or story."""
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    async with Client(graph.mcp) as client:
        with pytest.raises(ToolError, match="node_labels"):
            await client.call_tool("save_memory", {"relationships": [
                {"subject": "A", "relation": "KNOWS", "object": "B", "description": "A knows B"},
                {"subject": "B", "relation": "KNOWS", "object": "C", "description": "B knows C",
                 "object_type": "Invalid` label"},
            ], "story": "Rejected story"})
    records, _, _ = await graph.store.driver().execute_query("MATCH (n) RETURN count(n) AS count")
    assert records == [{"count": 0}]


@pytest.mark.asyncio
@pytest.mark.parametrize("provenance", [{}, {"story": "Unused", "source": "Unused"},
                                        {"story_id": "unused-episode"}])
async def test_empty_batches_remain_noops(graph, monkeypatch, provenance):
    """An empty batch neither creates nor looks up provenance and needs no services."""
    from unittest.mock import AsyncMock, Mock

    driver = Mock(side_effect=AssertionError("Empty batches must not access the graph"))
    initialize = AsyncMock()
    embedding = AsyncMock()
    monkeypatch.setattr(graph.store, "driver", driver)
    monkeypatch.setattr(graph.store, "ensure_indices", initialize)
    monkeypatch.setattr(graph.embed, "embed", embedding)
    assert await graph.save_memory(relationships=[], **provenance) == {
        "relationships": [], "story_id": provenance.get("story_id")}
    driver.assert_not_called()
    initialize.assert_not_awaited()
    embedding.assert_not_awaited()


# ── Stories ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_one_story_is_shared_by_a_batch(graph):
    """Facts from one conversation share a story rather than each carrying a
    copy — otherwise a four-hit search returns the same narrative four times."""
    r = await graph.save_memory(
        relationships=[_relationship(graph, "Max", "PREFERS", "a", "Max prefers a"),
               _relationship(graph, "Max", "PREFERS", "b", "Max prefers b")],
        story="## How this came up\nWe were designing the memory service.")
    assert r["story_id"]
    story = await graph.get_story(r["story_id"])
    assert story["relationship_count"] == 2
    assert "designing the memory service" in story["story"]
    assert {f["story_id"] for f in r["relationships"]} == {r["story_id"]}


@pytest.mark.asyncio
async def test_a_later_fact_can_join_an_existing_story(graph):
    """Adding to something already recorded should link, not re-narrate."""
    first = await graph.save_memory(
        relationships=[_relationship(graph, "Max", "PREFERS", "a", "Max prefers a")],
        story="The original conversation.")
    await graph.save_memory(
        relationships=[_relationship(graph, "Max", "PREFERS", "b", "Max prefers b")],
        story_id=first["story_id"])
    story = await graph.get_story(first["story_id"])
    assert story["relationship_count"] == 2


@pytest.mark.asyncio
async def test_facts_without_a_story_are_still_valid(graph):
    r = await graph.save_memory(relationships=[_relationship(graph, "Max", "PREFERS", "a",
                                            "Max prefers a")])
    assert r["story_id"] is None and r["relationships"][0]["relationship_id"]


@pytest.mark.asyncio
async def test_source_only_batches_keep_shared_provenance(graph):
    """Attribution survives saving and retrieval even without narrative text."""
    source = "archive/2026-05-12.md"
    result = await graph.save_memory(
        relationships=[_relationship(graph, "Max", "PREFERS", "a", "Max prefers a"),
               _relationship(graph, "Max", "PREFERS", "b", "Max prefers b")],
        source=source)
    assert result["story_id"]
    assert {fact["story_id"] for fact in result["relationships"]} == {result["story_id"]}
    story = await graph.get_story(result["story_id"])
    assert story["source"] == source
    assert story["story"] == ""
    assert story["relationship_count"] == 2
    hits = await graph.search(kind="relationships", query="Max")
    assert len(hits) == 2
    assert {hit["story_id"] for hit in hits} == {result["story_id"]}


@pytest.mark.asyncio
async def test_story_keeps_narrative_source_and_dates_separate(graph):
    """A historical source date must not be reported as the date it was saved."""
    when = datetime(2026, 5, 12, tzinfo=timezone.utc)
    result = await graph.save_memory(
        relationships=[_relationship(graph, "Max", "PREFERS", "a", "Max prefers a")],
        story="We discussed this in May.", source="archive/2026-05-12.md",
        valid_at=when)
    story = await graph.get_story(result["story_id"])
    assert story["story"] == "We discussed this in May."
    assert story["source"] == "archive/2026-05-12.md"
    assert datetime.fromisoformat(story["valid_at"]) == when
    assert datetime.fromisoformat(story["recorded_at"]) > when


@pytest.mark.asyncio
@pytest.mark.parametrize("provenance", [{"story": "ignored narrative"},
                                       {"source": "ignored source"}])
async def test_reusing_an_episode_rejects_conflicting_provenance(graph, provenance):
    """New attribution cannot be silently discarded when an episode is reused."""
    with pytest.raises(ValueError, match="story_id"):
        await graph.save_memory(
            relationships=[_relationship(graph, "Max", "PREFERS", "a", "Max prefers a")],
            story_id="existing-episode", **provenance)
    assert await graph.list_relationships() == {"relationships": [], "next_cursor": None}


# ── Time ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_superseding_keeps_the_old_fact(graph):
    """The reason for a graph over a key-value store: what was once true stays
    answerable."""
    r = await graph.save_memory(relationships=[_relationship(graph, "Nanhi", "RUNS_ON", "glm",
                                            "Nanhi runs on glm")])
    fid = r["relationships"][0]["relationship_id"]
    done = await graph.supersede(fid, reason="switched endpoints")
    assert done["superseded"] and done["invalid_at"]
    assert done["description"] == "Nanhi runs on glm"          # still there, verbatim

    entity = await graph.get_entity("Nanhi")
    assert entity["relationships"][0]["superseded"] is True


@pytest.mark.asyncio
async def test_a_backdated_fact_keeps_the_date_it_was_given(graph):
    """Recording something learned late must not claim it started now."""
    when = datetime.now(timezone.utc) - timedelta(days=900)
    r = await graph.save_memory(
        relationships=[_relationship(graph, "Max", "WORKS_AT", "somewhere", "Max works there")],
        valid_at=when)
    assert r["relationships"][0]["valid_at"].startswith(when.date().isoformat())


@pytest.mark.asyncio
async def test_each_fact_can_keep_its_own_historical_date(graph):
    """A mixed import retains individual dates and a fallback for undated entries."""
    april = datetime(2026, 4, 10, tzinfo=timezone.utc)
    july = datetime(2026, 7, 3, tzinfo=timezone.utc)
    fallback = datetime(2026, 6, 1, tzinfo=timezone.utc)
    first = _relationship(graph, "Max", "PLANNED", "a", "Max planned a in April")
    first.valid_at = april
    second = _relationship(graph, "Max", "FINISHED", "b", "Max finished b in July")
    second.valid_at = july
    third = _relationship(graph, "Max", "STARTED", "c", "Max started c in June")
    result = await graph.save_memory(relationships=[first, second, third], valid_at=fallback)
    assert [datetime.fromisoformat(f["valid_at"]) for f in result["relationships"]] == [
        april, july, fallback]
    assert all(datetime.fromisoformat(f["recorded_at"]) > july for f in result["relationships"])
    inventory = await graph.list_relationships()
    assert {f["relationship_id"]: f["valid_at"] for f in inventory["relationships"]} == {
        f["relationship_id"]: f["valid_at"] for f in result["relationships"]}


@pytest.mark.asyncio
async def test_explicit_unknown_date_does_not_become_the_import_date(graph):
    """Explicit null overrides the batch fallback and remains unknown on retrieval."""
    fact = _relationship(graph, "Max", "PLANNED", "a", "Max planned a sometime that spring")
    fact.valid_at = None
    result = await graph.save_memory(
        relationships=[fact], valid_at=datetime(2026, 5, 1, tzinfo=timezone.utc))
    assert result["relationships"][0]["valid_at"] is None
    assert result["relationships"][0]["recorded_at"]
    assert (await graph.search(kind="relationships", query="Max"))[0]["valid_at"] is None


@pytest.mark.asyncio
async def test_omitted_date_still_defaults_to_the_recording_time(graph, monkeypatch):
    """Existing callers retain the current-time default when they omit dates."""
    when = datetime(2026, 9, 12, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(graph.store, "now", lambda: when)
    result = await graph.save_memory(relationships=[
        _relationship(graph, "Max", "PREFERS", "a", "Max prefers a")])
    assert result["relationships"][0]["valid_at"] == result["relationships"][0]["recorded_at"] == when.isoformat()


@pytest.mark.asyncio
async def test_dates_without_a_timezone_are_interpreted_as_utc(graph):
    """Date-only imports receive consistent UTC timestamps on both date paths."""
    when = datetime(2026, 5, 1)
    first = _relationship(graph, "Max", "PLANNED", "a", "Max planned a")
    first.valid_at = when
    second = _relationship(graph, "Max", "PLANNED", "b", "Max planned b")
    result = await graph.save_memory(relationships=[first, second], valid_at=when)
    assert {fact["valid_at"] for fact in result["relationships"]} == {
        when.replace(tzinfo=timezone.utc).isoformat()}


@pytest.mark.asyncio
async def test_fact_dates_survive_mcp_tool_validation(graph):
    """The MCP boundary preserves historical, unknown, and omitted date inputs."""
    from fastmcp import Client

    base = {"subject": "Max", "relation": "PLANNED", "description": "Max made a plan"}
    async with Client(graph.mcp) as client:
        result = await client.call_tool("save_memory", {
            "relationships": [
                {**base, "object": "April", "valid_at": "2026-04-10T00:00:00Z"},
                {**base, "object": "unknown", "valid_at": None},
                {**base, "object": "fallback"},
            ],
            "valid_at": "2026-06-01T00:00:00Z",
        })
    relationships = result.structured_content["relationships"]
    assert [fact["valid_at"] for fact in relationships] == [
        "2026-04-10T00:00:00+00:00", None, "2026-06-01T00:00:00+00:00"]


@pytest.mark.asyncio
async def test_date_correction_retains_the_fact_story_and_embedding(embedded):
    """Repairing an import preserves identity, recording time, search, and provenance."""
    text = "Max planned a in May"
    result = await embedded.save_memory(
        relationships=[_relationship(embedded, "Max", "PLANNED", "a", text)],
        story="The dated source document.", source="archive/2026-05-01.md")
    original = result["relationships"][0]
    before = await embedded.EntityEdge.get_by_uuid(embedded.store.driver(), original["relationship_id"])
    await before.load_fact_embedding(embedded.store.driver())
    assert before.fact_embedding
    when = datetime(2026, 5, 1, tzinfo=timezone.utc)
    fixed = await embedded.set_valid_at(
        original["relationship_id"], when, reason="The import used the recording date")
    assert fixed["relationship_id"] == original["relationship_id"]
    assert fixed["description"] == text
    assert fixed["valid_at"] == when.isoformat()
    assert fixed["recorded_at"] == original["recorded_at"]
    assert fixed["story_id"] == result["story_id"]
    assert fixed["valid_at_corrections"][0]["previous_valid_at"] == original["valid_at"]
    assert fixed["valid_at_corrections"][0]["reason"] == "The import used the recording date"
    assert (await embedded.get_story(result["story_id"]))["relationship_count"] == 1
    assert (await embedded.search(kind="relationships", query=text))[0]["valid_at"] == when.isoformat()
    assert len((await embedded.list_relationships())["relationships"]) == 1
    repeated = await embedded.set_valid_at(original["relationship_id"], when)
    assert repeated["valid_at_corrections"] == fixed["valid_at_corrections"]
    unknown = await embedded.set_valid_at(original["relationship_id"], None, reason="Date uncertain")
    assert unknown["valid_at"] is None
    assert len(unknown["valid_at_corrections"]) == 2
    assert (await embedded.search(kind="relationships", query=text))[0]["valid_at"] is None
    restored = await embedded.set_valid_at(original["relationship_id"], when)
    assert restored["valid_at_corrections"][-1]["previous_valid_at"] is None
    after = await embedded.EntityEdge.get_by_uuid(embedded.store.driver(), original["relationship_id"])
    await after.load_fact_embedding(embedded.store.driver())
    assert after.fact_embedding == before.fact_embedding


@pytest.mark.asyncio
async def test_date_correction_cannot_invert_a_facts_validity(graph):
    """A correction cannot move a start date beyond the date the fact stopped applying."""
    original_date = datetime(2026, 4, 1, tzinfo=timezone.utc)
    result = await graph.save_memory(
        relationships=[_relationship(graph, "Max", "PLANNED", "a", "Max planned a")],
        valid_at=original_date)
    relationship_id = result["relationships"][0]["relationship_id"]
    await graph.supersede(relationship_id, invalid_at=datetime(2026, 5, 1, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="invalid_at"):
        await graph.set_valid_at(relationship_id, datetime(2026, 6, 1))
    stored = (await graph.list_relationships())["relationships"][0]
    assert stored["valid_at"] == original_date.isoformat()
    assert "valid_at_corrections" not in stored


@pytest.mark.asyncio
async def test_date_correction_cannot_modify_another_group(graph, monkeypatch):
    """A known ID is insufficient to modify a different memory group's date."""
    result = await graph.save_memory(relationships=[
        _relationship(graph, "Max", "PLANNED", "a", "Max planned a")])
    monkeypatch.setattr(graph.store, "GROUP_ID", "other")
    with pytest.raises(ValueError, match="group"):
        await graph.set_valid_at(result["relationships"][0]["relationship_id"], None)


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
    await embedded.save_memory(relationships=[_relationship(
        embedded, "Max", "PREFERS", "commit granularity",
        "Max prefers function-level commit granularity")])
    hits = await embedded.search(kind="relationships", query="Max prefers function-level commit granularity")
    assert hits and hits[0]["subject"] == "Max"
    assert hits[0]["relationship_id"]


@pytest.mark.asyncio
async def test_search_hides_superseded_facts_by_default(embedded):
    """Current picture by default; history only when asked for, so a stale
    fact cannot quietly present itself as true."""
    text = "Nanhi runs on glm"
    r = await embedded.save_memory(relationships=[_relationship(embedded, "Nanhi", "RUNS_ON", "glm", text)])
    await embedded.supersede(r["relationships"][0]["relationship_id"])

    assert await embedded.search(kind="relationships", query=text) == []
    history = await embedded.search(kind="relationships", query=text, include_superseded=True)
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

    old = await graph.save_memory(relationships=[
        _relationship(graph, "Heather", "PREFERS", "coffee", "Heather likes coffee")])
    old_id = old["relationships"][0]["relationship_id"]
    await graph.supersede(old_id)
    current = await graph.save_memory(relationships=[
        _relationship(graph, "Heather", "PREFERS", "tea",
              "Heather likes tea with breakfast every morning before starting work")])
    current_id = current["relationships"][0]["relationship_id"]
    query = "Heather" if mode == "keyword" else "preferred beverage"

    history = await graph.search(kind="relationships", query=query, limit=1, include_superseded=True)
    assert [hit["relationship_id"] for hit in history] == [old_id]
    hits = await graph.search(kind="relationships", query=query, limit=1)
    assert [hit["relationship_id"] for hit in hits] == [current_id]
    assert hits[0]["superseded"] is False


@pytest.mark.asyncio
async def test_search_returns_the_episode_id_so_linking_is_cheap(embedded):
    """Sharing a story has to be easier than rewriting one, or duplicates win
    on convenience alone."""
    text = "Max prefers function-level commits"
    await embedded.save_memory(relationships=[_relationship(embedded, "Max", "PREFERS", "commits", text)],
                              story="Where this came from.")
    hits = await embedded.search(kind="relationships", query=text)
    assert hits[0]["story_id"]
    story = await embedded.get_story(hits[0]["story_id"])
    assert "Where this came from" in story["story"]


@pytest.mark.asyncio
async def test_writes_still_work_with_no_embedding_service(graph):
    """An unreachable embedding endpoint must not cost the write — the fact is
    worth keeping without a vector and can be embedded later."""
    r = await graph.save_memory(relationships=[_relationship(graph, "Max", "PREFERS", "a",
                                            "Max prefers a")])
    assert r["relationships"][0]["relationship_id"]
    assert await graph.search(kind="relationships", query="anything") == []      # degrades, not errors
    assert (await graph.get_entity("Max"))["found"] is True


@pytest.mark.asyncio
async def test_unknown_entity_reports_not_found(graph):
    assert (await graph.get_entity("Nobody"))["found"] is False


@pytest.mark.asyncio
async def test_keyword_search_works_without_an_embedder(graph):
    """The reason this runs on FalkorDB rather than an embedded backend:
    fulltext indices are real here, so an exact name is findable even with no
    embedding service configured — the case vectors are worst at."""
    await graph.save_memory(relationships=[
        _relationship(graph, "Heather", "RUNS_ON", "glm-5.3", "Heather runs on glm-5.3",
              s_type="Agent", o_type="Model"),
        _relationship(graph, "Nanhi", "RUNS_ON", "kimi-k3", "Nanhi runs on kimi-k3",
              s_type="Agent", o_type="Model"),
    ])
    hits = await graph.search(kind="relationships", query="Heather")
    assert [h["subject"] for h in hits] == ["Heather"]


@pytest.mark.asyncio
async def test_search_merges_keyword_and_semantic_hits(embedded):
    """Each mode misses what the other catches: a paraphrase defeats keywords,
    an unseen proper noun defeats vectors. Results are the union, deduped."""
    await graph_save(embedded)
    hits = await embedded.search(kind="relationships", query="Heather", limit=10)
    assert any(h["subject"] == "Heather" for h in hits)
    ids = [h["relationship_id"] for h in hits]
    assert len(ids) == len(set(ids)), "a fact matched by both modes was returned twice"


@pytest.mark.asyncio
async def test_keyword_search_survives_a_vector_query_failure(embedded, monkeypatch):
    """A failing vector query must not prevent exact-name memory retrieval."""
    result = await embedded.save_memory(relationships=[
        _relationship(embedded, "Heather", "RUNS_ON", "model", "Heather runs on model")])

    async def unavailable(*args, **kwargs):
        """Simulate an unavailable vector index while the graph remains readable."""
        raise RuntimeError("Vector query unavailable")

    monkeypatch.setattr(embedded, "edge_similarity_search", unavailable)
    hits = await embedded.search(kind="relationships", query="Heather")
    assert [hit["relationship_id"] for hit in hits] == [result["relationships"][0]["relationship_id"]]


@pytest.mark.asyncio
async def test_semantic_threshold_can_be_tuned_for_the_embedding_model(graph, monkeypatch):
    """Model-specific score thresholds control semantic recall without keyword matches."""
    async def vector(text, *, is_query=False):
        """Return query/document vectors whose cosine similarity is one half."""
        return [1.0, 0.0] if is_query else [0.5, math.sqrt(0.75)]

    monkeypatch.setattr(graph.embed, "embed", vector)
    result = await graph.save_memory(relationships=[
        _relationship(graph, "Heather", "PREFERS", "tea", "Heather prefers tea")])
    monkeypatch.setattr(graph.embed, "MIN_SCORE", 0.8)
    assert await graph.search(kind="relationships", query="favored beverage") == []
    monkeypatch.setattr(graph.embed, "MIN_SCORE", 0.7)
    hits = await graph.search(kind="relationships", query="favored beverage")
    assert [hit["relationship_id"] for hit in hits] == [result["relationships"][0]["relationship_id"]]


async def graph_save(server):
    await server.save_memory(relationships=[
        _relationship(server, "Heather", "RUNS_ON", "glm-5.3", "Heather runs on glm-5.3",
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
    result = await graph.save_memory(relationships=[
        _relationship(graph, "Max", "RECORDED", f"item {i}", f"Max recorded item {i}")
        for i in range(5)])
    expected = sorted((fact["relationship_id"] for fact in result["relationships"]), reverse=True)
    await graph.supersede(expected[0])

    async def forbidden_embedding(text):
        """Fail if an inventory attempts to embed any text."""
        raise AssertionError("An inventory must not use embeddings")

    monkeypatch.setattr(graph.embed, "embed", forbidden_embedding)
    first = await graph.list_relationships(limit=2)
    second = await graph.list_relationships(limit=2, cursor=first["next_cursor"])
    third = await graph.list_relationships(limit=2, cursor=second["next_cursor"])
    pages = [first, second, third]
    assert [len(page["relationships"]) for page in pages] == [2, 2, 1]
    assert [fact["relationship_id"] for page in pages for fact in page["relationships"]] == expected
    assert first["relationships"][0]["superseded"] is True
    assert first["next_cursor"] == expected[1]
    assert second["next_cursor"] == expected[3]
    assert third["next_cursor"] is None
    assert await graph.list_relationships(cursor=expected[-1]) == {
        "relationships": [], "next_cursor": None}


@pytest.mark.asyncio
async def test_inventory_is_empty_for_an_empty_graph(graph):
    """An empty store produces a terminal empty page."""
    assert await graph.list_relationships() == {"relationships": [], "next_cursor": None}


@pytest.mark.asyncio
async def test_inventory_only_lists_the_configured_group(graph, monkeypatch):
    """An audit cannot enumerate relationships belonging to another memory group."""
    await graph.save_memory(relationships=[_relationship(graph, "Max", "KNOWS", "a", "Max knows a")])
    monkeypatch.setattr(graph.store, "GROUP_ID", "other")
    assert await graph.list_relationships() == {"relationships": [], "next_cursor": None}


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1, 101])
async def test_inventory_rejects_unbounded_page_sizes(graph, limit):
    """Page limits are bounded even when calling the implementation directly."""
    with pytest.raises(ValueError, match="limit"):
        await graph.list_relationships(limit=limit)


@pytest.mark.asyncio
async def test_inventory_rejects_invalid_cursors(graph):
    """Malformed cursors fail clearly instead of returning misleading pages."""
    with pytest.raises(ValueError):
        await graph.list_relationships(cursor="not-a-cursor")


async def _save_links(graph, links):
    """Store synthetic directed relationships and return their save response."""
    return await graph.save_memory(relationships=[
        graph.Relationship(subject=source, relation=relation, object=target,
                   description=f"{source} {relation} {target}")
        for source, relation, target in links])


@pytest.mark.asyncio
async def test_neighborhood_follows_multiple_hops_and_preserves_directions(graph):
    """A bounded neighborhood includes incoming links without reversing their meaning."""
    await _save_links(graph, [("A", "USES", "B"), ("C", "OWNS", "B"),
                              ("B", "NEEDS", "D"), ("D", "NEEDS", "E")])
    result = await graph.explore("A", max_hops=2)
    assert result["found"] and not result["truncated"]
    assert {node["name"]: node["hops"] for node in result["entities"]} == {
        "A": 0, "B": 1, "C": 2, "D": 2}
    assert {(fact["subject"], fact["object"]) for fact in result["relationships"]} == {
        ("A", "B"), ("C", "B"), ("B", "D")}
    ids = {node["entity_id"] for node in result["entities"]}
    assert all(fact["subject_id"] in ids and fact["object_id"] in ids for fact in result["relationships"])


@pytest.mark.asyncio
@pytest.mark.parametrize("direction,expected", [
    ("outgoing", {"B", "D"}), ("incoming", {"B", "A", "C"}),
])
async def test_neighborhood_direction_controls_expansion(graph, direction, expected):
    """Incoming and outgoing exploration follow different recorded relationships."""
    await _save_links(graph, [("A", "USES", "B"), ("C", "OWNS", "B"), ("B", "NEEDS", "D")])
    result = await graph.explore("B", max_hops=1, direction=direction)
    assert {node["name"] for node in result["entities"]} == expected


@pytest.mark.asyncio
async def test_traversal_survives_relationship_group_index_misses(graph, monkeypatch):
    """Indexed group misses must not hide a root or an intermediate entity's relationships."""
    from types import SimpleNamespace

    saved = await _save_links(graph, [("Max", "FRIEND_OF", "Nanhi"),
                                     ("Heather", "KNOWS", "Max"),
                                     ("Max", "OLD", "Former plan")])
    await graph.supersede(saved["relationships"][2]["relationship_id"])
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
    assert len((await graph.get_entity("Max"))["relationships"]) == 3
    neighborhood = await graph.explore("Max", max_hops=1)
    assert {fact["relationship_id"] for fact in neighborhood["relationships"]} == {
        fact["relationship_id"] for fact in saved["relationships"][:2]}
    assert not neighborhood["truncated"]
    for direction, other in [("outgoing", "Nanhi"), ("incoming", "Heather")]:
        result = await graph.explore("Max", max_hops=1, direction=direction)
        assert {node["name"] for node in result["entities"]} == {"Max", other}
    historical = await graph.explore("Max", max_hops=1, include_superseded=True)
    assert len(historical["relationships"]) == 3
    for source, target, hops in [("Max", "Nanhi", 1), ("Heather", "Nanhi", 2)]:
        path = await graph.find_path(source, target, directed=True)
        assert path["found"] and path["hops"] == hops
        assert all(fact["traversed_forward"] for fact in path["relationships"])
    assert (await graph.find_path("Nanhi", "Heather"))["hops"] == 2


@pytest.mark.asyncio
async def test_exploration_handles_cycles_self_links_and_parallel_facts(graph, monkeypatch):
    """Cycles do not duplicate relationships or trigger embedding calls during graph reads."""
    saved = await _save_links(graph, [("A", "LIKES", "B"), ("A", "KNOWS", "B"),
                                     ("B", "KNOWS", "C"), ("C", "KNOWS", "A"),
                                     ("B", "REFLECTS_ON", "B")])

    async def forbidden(*args, **kwargs):
        """Fail if graph-only exploration attempts to call an embedding model."""
        raise AssertionError("Graph exploration must not embed text")

    monkeypatch.setattr(graph.embed, "embed", forbidden)
    neighborhood = await graph.explore("A", max_hops=6)
    assert len(neighborhood["entities"]) == 3
    assert {fact["relationship_id"] for fact in neighborhood["relationships"]} == {
        fact["relationship_id"] for fact in saved["relationships"]}
    assert not neighborhood["truncated"]
    path = await graph.find_path("A", "B")
    assert path["hops"] == 1 and len(path["relationships"]) == 1
    assert (await graph.explain_relationship(path["relationships"][0]["relationship_id"]))["found"]
    assert len((await graph.list_relationships())["relationships"]) == 5


@pytest.mark.asyncio
async def test_superseded_links_cannot_bridge_a_current_exploration(graph):
    """Filtering happens before traversal, including when the current route is longer."""
    saved = await _save_links(graph, [("A", "OLD", "B"), ("B", "NEXT", "C"),
                                     ("A", "NEW", "D"), ("D", "NEXT", "E"),
                                     ("E", "NEXT", "C")])
    await graph.supersede(saved["relationships"][0]["relationship_id"])
    neighborhood = await graph.explore("A", max_hops=2)
    assert {node["name"] for node in neighborhood["entities"]} == {"A", "D", "E"}
    current = await graph.find_path("A", "C", max_hops=3)
    assert [node["name"] for node in current["entities"]] == ["A", "D", "E", "C"]
    assert all(not fact["superseded"] for fact in current["relationships"])
    historical = await graph.find_path("A", "C", include_superseded=True)
    assert [node["name"] for node in historical["entities"]] == ["A", "B", "C"]
    assert historical["relationships"][0]["superseded"]


@pytest.mark.asyncio
async def test_path_reports_reverse_traversal_without_reversing_the_fact(graph):
    """An undirected connection is not a newly asserted reverse relationship."""
    await _save_links(graph, [("A", "OWNS", "B")])
    reverse = await graph.find_path("B", "A")
    assert reverse["found"] and reverse["hops"] == 1
    fact = reverse["relationships"][0]
    assert (fact["subject"], fact["relation"], fact["object"]) == ("A", "OWNS", "B")
    assert fact["traversed_forward"] is False
    assert fact["from_entity_id"] == fact["object_id"]
    assert fact["to_entity_id"] == fact["subject_id"]
    assert not (await graph.find_path("B", "A", directed=True))["found"]
    assert (await graph.find_path("A", "B", directed=True))["relationships"][0]["traversed_forward"]


@pytest.mark.asyncio
async def test_path_prefers_the_shortest_eligible_route(graph):
    """The result uses the shorter of two eligible routes and respects the hop cap."""
    await _save_links(graph, [("A", "NEXT", "B"), ("B", "NEXT", "D"),
                              ("A", "NEXT", "C"), ("C", "NEXT", "E"), ("E", "NEXT", "D")])
    result = await graph.find_path("A", "D", directed=True)
    assert [node["name"] for node in result["entities"]] == ["A", "B", "D"]
    assert result["hops"] == 2
    too_short = await graph.find_path("A", "D", max_hops=1)
    assert not too_short["found"] and too_short["max_hops"] == 1


@pytest.mark.asyncio
async def test_fact_budget_reports_incomplete_search_and_exact_completion(graph):
    """The budget flag distinguishes an unexplored frontier from a complete small graph."""
    await _save_links(graph, [("A", "NEXT", "B"), ("B", "NEXT", "C")])
    limited = await graph.explore("A", max_hops=2, limit=1)
    assert len(limited["relationships"]) == 1 and limited["truncated"]
    radius_one = await graph.explore("A", max_hops=1, limit=1)
    assert not radius_one["truncated"]
    complete = await graph.explore("A", max_hops=3, limit=2)
    assert len(complete["relationships"]) == 2 and not complete["truncated"]
    path = await graph.find_path("A", "C", limit=1)
    assert not path["found"] and path["truncated"] and path["explored_relationships"] == 1


@pytest.mark.asyncio
async def test_direct_target_is_prioritized_with_a_small_fact_budget(graph):
    """A direct target remains discoverable even when other adjacent relationships exceed the budget."""
    await _save_links(graph, [("A", "KNOWS", "B"), ("A", "KNOWS", "C"), ("A", "KNOWS", "D")])
    path = await graph.find_path("A", "D", limit=1)
    assert path["found"] and path["hops"] == 1
    assert path["explored_relationships"] == 1 and path["truncated"]


@pytest.mark.asyncio
async def test_exploration_handles_missing_isolated_and_identical_entities(graph):
    """Missing entities are explicit, while a known entity has a zero-hop path to itself."""
    assert not (await graph.explore("missing"))["found"]
    missing = await graph.find_path("missing", "also missing")
    assert missing["missing_entities"] == ["missing", "also missing"]
    await graph.store.upsert_entity("A", "Thing")
    isolated = await graph.explore("A")
    assert isolated["found"] and isolated["relationships"] == [] and not isolated["truncated"]
    assert isolated["entities"][0]["hops"] == 0
    same = await graph.find_path("A", "A")
    assert same["found"] and same["hops"] == 0 and same["relationships"] == []


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
    result = await graph.explore("A", max_hops=6)
    assert {node["name"] for node in result["entities"]} == {"A", "B"}
    assert (await graph.find_path("A", "X"))["missing_entities"] == ["X"]
    with pytest.raises(ValueError, match="group"):
        await graph.explain_relationship(foreign_facts["relationships"][0]["relationship_id"])
    with pytest.raises(RuntimeError):
        await graph.explain_relationship(cross_group.uuid)


@pytest.mark.asyncio
@pytest.mark.parametrize("method,arguments", [
    ("explore", {"name": "A", "max_hops": 0}),
    ("explore", {"name": "A", "max_hops": 7}),
    ("explore", {"name": "A", "limit": 0}),
    ("explore", {"name": "A", "limit": 101}),
    ("explore", {"name": "A", "direction": "sideways"}),
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
    saved = await graph.save_memory(relationships=[graph.Relationship(
        subject="A", relation="PLANS", object="B", description="A appears to be planning B",
        evidence_kind="inferred", rationale="A requested components used by B")],
        story="The original conversation supplied the context.", source="messages/123")
    relationship_id = saved["relationships"][0]["relationship_id"]
    await graph.set_valid_at(relationship_id, datetime(2026, 5, 1, tzinfo=timezone.utc), reason="Source date")
    await graph.supersede(relationship_id, reason="The plan was cancelled")
    result = await graph.explain_relationship(relationship_id)
    assert result["found"]
    assert result["relationship"]["evidence_kind"] == "inferred"
    assert result["relationship"]["rationale"] == "A requested components used by B"
    assert result["relationship"]["valid_at_corrections"][0]["reason"] == "Source date"
    assert result["superseded_reason"] == "The plan was cancelled"
    assert result["sources"][0]["source"] == "messages/123"
    assert result["sources"][0]["story"] == "The original conversation supplied the context."
    assert result["missing_story_ids"] == [] and not result["sources_truncated"]


@pytest.mark.asyncio
async def test_explanation_bounds_stories_and_preserves_source_only_attribution(graph):
    """Narrative and attribution truncation are explicit and full stories remain readable."""
    saved = await graph.save_memory(relationships=[_relationship(graph, "A", "KNOWS", "B", "A knows B")],
                                   story="Long narrative", source="s" * 1001)
    result = await graph.explain_relationship(saved["relationships"][0]["relationship_id"], story_chars=4)
    source = result["sources"][0]
    assert source["story"] == "Long" and source["story_truncated"]
    assert len(source["source"]) == 1000 and source["source_truncated"]
    assert (await graph.get_story(source["story_id"]))["story"] == "Long narrative"
    source_only = await graph.save_memory(
        relationships=[_relationship(graph, "A", "KNOWS", "C", "A knows C")], source="messages/456")
    provenance = (await graph.explain_relationship(source_only["relationships"][0]["relationship_id"]))["sources"][0]
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
    edge = await graph.EntityEdge.get_by_uuid(graph.store.driver(), saved["relationships"][0]["relationship_id"])
    missing_id = str(uuid.uuid4())
    edge.episodes = [first.uuid, first.uuid, missing_id, second.uuid, foreign.uuid]
    await graph.store.update_edge(edge)
    limited = await graph.explain_relationship(edge.uuid, source_limit=2, story_chars=0)
    assert len(limited["sources"]) == 1 and limited["sources_truncated"]
    assert limited["missing_story_ids"] == [missing_id]
    assert limited["sources"][0]["story_truncated"] and limited["sources"][0]["story"] == ""
    complete = await graph.explain_relationship(edge.uuid, source_limit=10)
    assert len(complete["sources"]) == 2 and not complete["sources_truncated"]
    assert complete["missing_story_ids"] == [missing_id, foreign.uuid]


@pytest.mark.asyncio
async def test_explanation_does_not_assign_evidence_to_legacy_facts(graph):
    """Legacy records with no evidence metadata remain explicitly unspecified."""
    saved = await _save_links(graph, [("A", "KNOWS", "B")])
    edge = await graph.EntityEdge.get_by_uuid(graph.store.driver(), saved["relationships"][0]["relationship_id"])
    edge.attributes = {}
    await graph.store.update_edge(edge)
    result = await graph.explain_relationship(edge.uuid)
    assert result["relationship"]["evidence_kind"] == "unspecified"
    assert result["relationship"]["rationale"] == "" and result["sources"] == []
    assert not (await graph.explain_relationship(str(uuid.uuid4())))["found"]


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [
    {"source_limit": 0}, {"source_limit": 11}, {"story_chars": -1}, {"story_chars": 5001},
])
async def test_explanation_rejects_unbounded_settings(graph, arguments):
    """Explanation limits are validated before loading a fact."""
    with pytest.raises(ValueError):
        await graph.explain_relationship(str(uuid.uuid4()), **arguments)


@pytest.mark.asyncio
@pytest.mark.parametrize("method,arguments", [
    ("explore", {"name": "A"}),
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
        for name in ["explore", "find_path", "explain_relationship"]:
            assert tools[name].annotations.readOnlyHint is True
        assert tools["explore"].inputSchema["properties"]["max_hops"]["maximum"] == 6
        saved = await client.call_tool("save_memory", {"relationships": [{
            "subject": "A", "relation": "KNOWS", "object": "B", "description": "A knows B",
            "evidence_kind": "reported", "rationale": "A said so",
        }]})
        neighborhood = await client.call_tool("explore", {"name": "A"})
        assert neighborhood.structured_content["relationships"][0]["evidence_kind"] == "reported"
        path = await client.call_tool("find_path", {"source": "A", "target": "B"})
        assert path.structured_content["hops"] == 1
        explained = await client.call_tool("explain_relationship", {
            "relationship_id": saved.structured_content["relationships"][0]["relationship_id"],
        })
        assert explained.structured_content["relationship"]["rationale"] == "A said so"


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
        await graph.explain_relationship(str(uuid.uuid4()))


@pytest.mark.asyncio
async def test_exploration_does_not_return_a_broken_node_snapshot(graph, monkeypatch):
    """Disappearing entities cause a retryable failure, not a malformed explanation graph."""
    await _save_links(graph, [("A", "KNOWS", "B")])

    async def missing_nodes(*args, **kwargs):
        """Simulate entities disappearing between frontier and result reads."""
        return []

    monkeypatch.setattr(graph.EntityNode, "get_by_uuids", missing_nodes)
    with pytest.raises(RuntimeError, match="Graph changed"):
        await graph.explore("A")


def test_evidence_kind_rejects_unrecognized_certainty_labels():
    """Evidence categories cannot silently become arbitrary confidence claims."""
    from graphiti_mcp.server import Relationship

    with pytest.raises(ValueError):
        Relationship(subject="A", relation="KNOWS", object="B", description="A knows B", evidence_kind="verified")


@pytest.mark.asyncio
async def test_three_todos_remain_mutable_when_indexed_uuid_reads_miss_them(graph, monkeypatch):
    """Search-visible relationships can be corrected without indexed UUID reads or edge upserts."""
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
    saved = await graph.save_memory(relationships=[
        _relationship(graph, "Max", "TODO", f"task {i}", f"Max has task {i} to do",
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
    assert {hit["relationship_id"] for hit in await graph.search(kind="relationships", query="Max")} == set(identifiers)

    when = datetime(2026, 5, 1, tzinfo=timezone.utc)
    for original in saved["relationships"]:
        fixed = await graph.set_valid_at(original["relationship_id"], when, reason="Source date")
        assert fixed["valid_at"] == when.isoformat()
        assert fixed["recorded_at"] == original["recorded_at"]
        assert fixed["description"] == original["description"]
        assert (await graph.explain_relationship(original["relationship_id"]))["found"]
    assert (await graph.supersede(identifiers[0], reason="Completed"))["superseded"]

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
    identifier = saved["relationships"][0]["relationship_id"]
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
    identifier = saved["relationships"][0]["relationship_id"]
    edge = await graph.store.get_edge(identifier)
    original_date = edge.valid_at
    await graph.store.driver().execute_query(
        "MATCH (s:Entity)-[e:RELATES_TO]->(t:Entity) WHERE toString(e.uuid) = $uuid "
        "CREATE (s)-[duplicate:RELATES_TO]->(t) SET duplicate = properties(e)", uuid=identifier)
    with pytest.raises(ValueError, match="Multiple relationships"):
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
    edge = await graph.store.get_edge(saved["relationships"][0]["relationship_id"])
    if change == "missing":
        edge.uuid = str(uuid.uuid4())
    elif change == "endpoints":
        edge.target_node_uuid = edge.source_node_uuid
    else:
        edge.group_id = "other"
    with pytest.raises(ValueError if change == "group" else RuntimeError):
        await graph.store.update_edge(edge)
    relationships = (await graph.list_relationships())["relationships"]
    assert len(relationships) == 1
    assert relationships[0]["relationship_id"] == saved["relationships"][0]["relationship_id"]
    assert (relationships[0]["subject"], relationships[0]["object"]) == ("A", "B")


@pytest.mark.asyncio
async def test_entity_inventory_pages_actual_entities_including_isolated_ones(graph, monkeypatch):
    """Entity pagination lists scoped nodes once, not their relationships or story records."""
    await graph.store.ensure_indices()
    identifiers = [
        "f655d1e4-03ce-4a4f-a9b7-9104a03bfb95",
        "f4678cfd-1682-4e4d-ab41-f887b80776fb",
        "4e17917b-4ae8-4360-8d14-d3a1ad91c50e",
        "13e2080d-0c6f-4f00-9bfe-7abc557b1af9",
        "031ded59-004f-4b69-b906-607810a493e2",
    ]
    for index, identifier in enumerate(identifiers):
        await graph.EntityNode(
            uuid=identifier, name=f"Entity {index}", group_id=graph.store.GROUP_ID,
            labels=["Entity", "Topic"], created_at=graph.store.now(),
        ).save(graph.store.driver())
    await graph.store.save_episode("A story is not an entity")
    with monkeypatch.context() as scoped:
        scoped.setattr(graph.store, "GROUP_ID", "other")
        await graph.store.upsert_entity("Foreign", "Person")
    first = await graph.list_entities(limit=2)
    second = await graph.list_entities(limit=2, cursor=first["next_cursor"])
    last = await graph.list_entities(limit=2, cursor=second["next_cursor"])
    assert [entity["entity_id"] for page in [first, second, last]
            for entity in page["entities"]] == identifiers
    assert first["entities"][0] == {
        "entity_id": identifiers[0], "name": "Entity 0", "types": ["Topic"]}
    assert first["next_cursor"] == identifiers[1]
    assert second["next_cursor"] == identifiers[3]
    assert last["next_cursor"] is None
    assert await graph.list_entities(cursor=identifiers[-1]) == {
        "entities": [], "next_cursor": None}


@pytest.mark.asyncio
async def test_entity_inventory_is_empty_for_an_empty_graph(graph):
    """An empty entity inventory has no cursor and does not invent placeholder records."""
    assert await graph.list_entities() == {"entities": [], "next_cursor": None}


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [{"limit": 0}, {"limit": 101}, {"cursor": "invalid"}])
async def test_entity_inventory_validates_bounds_and_cursor(graph, arguments):
    """Entity inventories reject invalid input even when called without MCP validation."""
    with pytest.raises(ValueError):
        await graph.list_entities(**arguments)


@pytest.mark.asyncio
async def test_entity_search_returns_names_with_bounded_relationship_previews(graph):
    """Search defaults to entity nodes, while relationship previews carry story links."""
    saved = await graph.save_memory(relationships=[
        _relationship(graph, "Max", "FEELS", "uneasy", "Max seemed uneasy; I could be misreading it"),
        _relationship(graph, "Max", "RECALLS", "home", "Max remembers home fondly"),
    ], story="The conversation felt hesitant. These are impressions, not verified conclusions.")
    entity = (await graph.search("Max", relationship_limit=1))[0]
    assert entity["name"] == "Max" and entity["entity_id"]
    assert len(entity["relationships"]) == 1 and entity["relationships_truncated"]
    assert entity["relationships"][0]["story_id"] == saved["story_id"]
    assert "relationship_id" not in entity
    complete = (await graph.search("Max"))[0]
    assert len(complete["relationships"]) == 2 and not complete["relationships_truncated"]
    assert (await graph.get_entity("Max"))["entity_id"] == entity["entity_id"]
    relationship_hits = await graph.search("uneasy", kind="relationships")
    assert len(relationship_hits) == 1
    assert "misreading" in relationship_hits[0]["description"]


@pytest.mark.asyncio
async def test_entity_search_finds_keywords_without_exact_name_or_embeddings(graph):
    """Partial-name keyword search returns actual entities, including isolated ones."""
    await graph.store.ensure_indices()
    node = await graph.store.upsert_entity("Copper Workshop", "Place")
    hits = await graph.search("Copper")
    assert [hit["entity_id"] for hit in hits] == [node.uuid]
    assert hits[0]["relationships"] == [] and not hits[0]["relationships_truncated"]


@pytest.mark.asyncio
async def test_entity_search_prioritizes_exact_names_and_merges_other_matches(embedded):
    """An exact name stays first and an entity matched by multiple modes appears once."""
    await _save_links(embedded, [("Max", "KNOWS", "Nanhi")])
    hits = await embedded.search("Max")
    assert hits[0]["name"] == "Max"
    assert len({hit["entity_id"] for hit in hits}) == len(hits)


@pytest.mark.asyncio
async def test_entity_search_semantics_use_name_embeddings_and_query_mode(graph, monkeypatch):
    """Entity vectors support paraphrases without relying on relationship-description vectors."""
    calls = []

    async def embedding(text, *, is_query=False):
        """Match the subject's name and a paraphrase with a deterministic vector."""
        calls.append((text, is_query))
        return [1.0, 0.0] if text in {"Workshop", "place for repairs"} else [-1.0, 0.0]

    monkeypatch.setattr(graph.embed, "embed", embedding)
    await _save_links(graph, [("Workshop", "CONTAINS", "Tools")])
    hits = await graph.search("place for repairs")
    assert [hit["name"] for hit in hits] == ["Workshop"]
    assert calls[-1] == ("place for repairs", True)


@pytest.mark.asyncio
async def test_entity_search_failures_do_not_hide_exact_names(embedded, monkeypatch):
    """Exact lookup remains available when both optional retrieval modes fail."""
    from unittest.mock import AsyncMock

    await _save_links(embedded, [("Max", "KNOWS", "Nanhi")])
    monkeypatch.setattr(embedded, "node_similarity_search", AsyncMock(side_effect=RuntimeError("vector")))
    monkeypatch.setattr(embedded, "node_fulltext_search", AsyncMock(side_effect=RuntimeError("keyword")))
    assert [hit["name"] for hit in await embedded.search("Max")] == ["Max"]


@pytest.mark.asyncio
async def test_entity_search_filters_relationship_history_not_the_entity(graph, monkeypatch):
    """Entities remain discoverable after supersession, without exposing another group's nodes."""
    saved = await _save_links(graph, [("Max", "KNOWS", "Nanhi")])
    await graph.supersede(saved["relationships"][0]["relationship_id"])
    assert (await graph.search("Max"))[0]["relationships"] == []
    assert (await graph.search("Max", include_superseded=True))[0]["relationships"][0]["superseded"]
    monkeypatch.setattr(graph.store, "GROUP_ID", "other")
    assert await graph.search("Max") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [
    {"limit": 0}, {"limit": 101}, {"kind": "facts"},
    {"relationship_limit": 0}, {"relationship_limit": 21},
])
async def test_search_validates_kind_and_bounds(graph, arguments):
    """Invalid search modes and unbounded results fail before reading or embedding."""
    with pytest.raises(ValueError):
        await graph.search("Max", **arguments)


@pytest.mark.asyncio
async def test_entity_explanation_preserves_sources_reasoning_and_history(graph):
    """Entity explanations collect stored relationship context, never generate conclusions."""
    saved = await graph.save_memory(relationships=[graph.Relationship(
        subject="Max", relation="FEELS", object="uneasy", description="Max seemed uneasy",
        evidence_kind="uncertain", rationale="My impression from the conversation",
    )], story="He paused before answering.", source="message/123")
    relationship_id = saved["relationships"][0]["relationship_id"]
    await graph.set_valid_at(relationship_id, None, reason="Date was uncertain")
    result = await graph.explain_entity("Max", story_chars=2)
    assert result["found"] and result["entity_id"] == saved["relationships"][0]["subject_id"]
    explanation = result["relationships"][0]
    assert explanation["relationship"]["evidence_kind"] == "uncertain"
    assert explanation["relationship"]["valid_at_corrections"][0]["reason"] == "Date was uncertain"
    assert explanation["sources"][0]["story"] == "He"
    assert explanation["sources"][0]["story_truncated"]
    assert explanation["sources"][0]["story_id"] == saved["story_id"]
    await graph.supersede(relationship_id, reason="My interpretation changed")
    assert (await graph.explain_entity("Max"))["relationships"] == []
    historical = await graph.explain_entity("Max", include_superseded=True)
    assert historical["relationships"][0]["superseded_reason"] == "My interpretation changed"


@pytest.mark.asyncio
async def test_entity_explanation_handles_missing_isolated_and_truncated_results(graph):
    """A missing entity differs from one with no relationships or an incomplete explanation."""
    assert await graph.explain_entity("missing") == {
        "found": False, "name": "missing", "relationships": [], "truncated": False}
    await graph.store.upsert_entity("Isolated", "Topic")
    isolated = await graph.explain_entity("Isolated")
    assert isolated["found"] and isolated["relationships"] == [] and not isolated["truncated"]
    await _save_links(graph, [("A", "KNOWS", "B"), ("A", "KNOWS", "C")])
    limited = await graph.explain_entity("A", limit=1)
    assert limited["truncated"] and len(limited["relationships"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [
    {"limit": 0}, {"limit": 21}, {"source_limit": 0}, {"source_limit": 11},
    {"story_chars": -1}, {"story_chars": 5001},
])
async def test_entity_explanation_validates_limits(graph, arguments):
    """Invalid explanation settings cannot produce unbounded reads."""
    with pytest.raises(ValueError):
        await graph.explain_entity("Max", **arguments)


@pytest.mark.asyncio
async def test_renamed_api_reads_existing_storage_without_migration(graph):
    """Legacy Graphiti properties retain IDs, embeddings, story links, and date history."""
    import json

    subject = await graph.store.upsert_entity("Max", "Person")
    obj = await graph.store.upsert_entity("Workshop", "Place")
    episode = await graph.store.save_episode("Original narrative", "original-source")
    edge = await graph.store.save_edge(
        subject, "RECALLS", obj, "Max remembers the Workshop", episode_uuid=episode.uuid,
        embedding=[0.6, 0.8], attributes={"valid_at_corrections": json.dumps([{
            "valid_at": None, "reason": "Original correction", "corrected_at": "2026-05-01",
        }])})
    await graph.store.link_episode(episode, [edge.uuid])
    result = await graph.explain_relationship(edge.uuid)
    relationship = result["relationship"]
    assert relationship["relationship_id"] == edge.uuid
    assert relationship["description"] == edge.fact
    assert relationship["story_id"] == episode.uuid
    assert relationship["valid_at_corrections"][0]["reason"] == "Original correction"
    assert relationship["evidence_kind"] == "unspecified"
    assert (await graph.get_story(episode.uuid))["relationship_count"] == 1
    properties, _, _ = await graph.store.driver().execute_query(
        "MATCH ()-[e:RELATES_TO]->() RETURN e.fact AS sentence, e.episodes AS episodes, "
        "e.fact_embedding AS embedding")
    assert properties == [{"sentence": edge.fact, "episodes": [episode.uuid],
                           "embedding": pytest.approx([0.6, 0.8])}]


@pytest.mark.asyncio
async def test_mcp_terminology_schema_and_entity_workflow(graph):
    """MCP advertises only the new vocabulary and every documented tool actually exists."""
    import json
    import re
    from fastmcp import Client

    async with Client(graph.mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        assert set(tools) == {
            "save_memory", "search", "list_entities", "list_relationships", "explore",
            "find_path", "explain_entity", "explain_relationship", "supersede",
            "set_valid_at", "get_story", "get_entity", "list_vocabulary",
        }
        assert tools["supersede"].inputSchema["required"] == ["relationship_id"]
        assert tools["search"].inputSchema["properties"]["kind"]["default"] == "entities"
        for name in set(tools) - {"save_memory", "supersede", "set_valid_at"}:
            assert tools[name].annotations.readOnlyHint
        schema = json.dumps(tools["save_memory"].inputSchema)
        relationship_schema = tools["save_memory"].inputSchema["properties"]["relationships"]["items"]
        assert "description" in relationship_schema["required"]
        assert not any(f'"{old}"' in schema for old in ["Fact", "Concept", "fact", "facts", "episode_id"])
        saved = await client.call_tool("save_memory", {
            "relationships": [{"subject": "A", "relation": "FEELS", "object": "B",
                               "description": "A has an uncertain impression of B"}],
            "story": "A subjective memory, without an objective claim.",
        })
        relationship = saved.structured_content["relationships"][0]
        inventory = await client.call_tool("list_entities", {"limit": 1})
        assert len(inventory.structured_content["entities"]) == 1
        assert inventory.structured_content["next_cursor"]
        explained = await client.call_tool("explain_entity", {"name": "A"})
        assert explained.structured_content["relationships"][0]["relationship"] == relationship
        corrected = await client.call_tool("set_valid_at", {
            "relationship_id": relationship["relationship_id"], "valid_at": None})
        assert corrected.structured_content["valid_at"] is None
        await client.call_tool("supersede", {"relationship_id": relationship["relationship_id"]})
        text = json.dumps(saved.structured_content)
        assert not any(f'"{old}"' in text for old in ["fact", "facts", "fact_id", "episode_id"])
    assert "The *story* is the memory" in graph.mcp.instructions
    assert "feelings" in graph.mcp.instructions and "not objective fact" in graph.mcp.instructions
    documented = re.findall(r"`([a-z_]+)(?:\([^`]*\))?`", graph.mcp.instructions)
    parameters = {"valid_at", "recorded_at", "truncated", "source", "story_id"}
    assert set(documented) - parameters <= set(tools)
