"""Graph storage layer — graphiti-core driven directly, no extraction pipeline.

graphiti's `add_episode` runs an LLM over raw text to invent entities and
relationships. We don't want that: the agent writing the memory is already a
capable model, and an extraction call per write costs a round trip on the same
endpoint she's using to think. Underneath that pipeline is an ordinary CRUD
layer — `EntityNode`/`EntityEdge`/`EpisodicNode` with `save()` — and that is
what this module uses.

The graph is stored in FalkorDB. An embedded backend would be lighter — Kuzu
was the obvious pick and graphiti still ships a driver for it — but Kùzu Inc.
was acquired in October 2025, the repository is archived, and graphiti's own
driver now warns that the backend is deprecated and slated for removal. A dead
database is not a foundation for long-term memory, so we pay for a service.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Literal

from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.edges import EntityEdge, get_entity_edge_from_record
from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.helpers import validate_node_labels
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_return_query
from graphiti_core.models.nodes.node_db_queries import get_entity_node_return_query
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode, get_entity_node_from_record
from redis.exceptions import ResponseError

Direction = Literal["both", "outgoing", "incoming"]

GRAPH_HOST = os.environ.get("GRAPH_HOST", "localhost")
GRAPH_PORT = int(os.environ.get("GRAPH_PORT", "6379"))
GRAPH_DATABASE = os.environ.get("GRAPH_DATABASE", "autonome")
GROUP_ID = os.environ.get("GRAPH_GROUP_ID", "main")

_driver: FalkorDriver | None = None


def driver() -> FalkorDriver:
    """The graph connection, opened on first use."""
    global _driver
    if _driver is None:
        _driver = FalkorDriver(host=GRAPH_HOST, port=GRAPH_PORT,
                               database=GRAPH_DATABASE)
    return _driver


_indices_ready = False


async def ensure_indices() -> None:
    """Create graphiti's indices, including the fulltext ones. Idempotent.

    Unlike the Kuzu driver — where this is a no-op and the fulltext path is
    consequently dead — FalkorDB builds real indices here, which is what makes
    keyword search work alongside vectors.
    """
    global _indices_ready
    if _indices_ready:
        return
    await driver().build_indices_and_constraints()
    _indices_ready = True


def now() -> datetime:
    return datetime.now(timezone.utc)


def utc_time(value: datetime) -> datetime:
    """Normalize a supplied date to UTC.

    Args:
        value: A timestamp. A missing timezone is interpreted as UTC.

    Returns:
        The equivalent timezone-aware UTC timestamp.
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


async def find_entity(name: str) -> EntityNode | None:
    """Look up an entity by exact name.

    graphiti exposes no name lookup — only uuid and group — so this is a
    direct query. Without it every save would mint a second `Max`.
    """
    records, _, _ = await driver().execute_query(
        "MATCH (n:Entity) WHERE n.name = $name AND n.group_id = $group_id "
        "RETURN n.uuid AS uuid LIMIT 1",
        name=name, group_id=GROUP_ID,
    )
    if not records:
        return None
    return await EntityNode.get_by_uuid(driver(), records[0]["uuid"])


def normalize_entity_type(entity_type: str) -> str:
    """Normalize and validate an optional entity type without accessing storage.

    Args:
        entity_type: A type label; whitespace is collapsed to underscores.

    Returns:
        A safe graph label, or an empty string for a blank type.

    Raises:
        ValueError: If the normalized type is not a valid graph label.
    """
    label = "_".join(entity_type.split())
    validate_node_labels([label] if label else [])
    return label


async def upsert_entity(name: str, entity_type: str, summary: str = "",
                        attributes: dict[str, Any] | None = None,
                        embedding: list[float] | None = None) -> EntityNode:
    """Fetch an entity by name, or create it.

    An existing entity keeps its identity: a later mention adds the type and
    fills an empty summary rather than replacing what is already known.

    Args:
        name: Exact entity name used for lookup.
        entity_type: Optional type. Surrounding whitespace is trimmed and
            internal whitespace becomes underscores, so 'Chat room' and
            'Chat_room' share one type. Blank types are omitted.
        summary: Description to set on creation or when the existing one is empty.
        attributes: Properties to merge into the entity.
        embedding: Name embedding for a newly created entity.

    Returns:
        The stored entity with its canonical type labels.

    Raises:
        ValueError: If the normalized type is not a valid graph label.
    """
    entity_type = normalize_entity_type(entity_type)
    existing = await find_entity(name)
    if existing is not None:
        changed = False
        if entity_type and entity_type not in existing.labels:
            existing.labels = list(existing.labels) + [entity_type]
            changed = True
        if summary and not existing.summary:
            existing.summary = summary
            changed = True
        if attributes:
            existing.attributes = {**(existing.attributes or {}), **attributes}
            changed = True
        if changed:
            await existing.save(driver())
        return existing

    node = EntityNode(
        name=name,
        group_id=GROUP_ID,
        labels=["Entity"] + ([entity_type] if entity_type else []),
        summary=summary,
        created_at=now(),
        attributes=attributes or {},
        name_embedding=embedding,
    )
    await node.save(driver())
    return node


async def save_episode(content: str, description: str = "",
                       valid_at: datetime | None = None) -> EpisodicNode:
    """Store the narrative a set of facts came from."""
    ep = EpisodicNode(
        name=(content.strip().splitlines() or [""])[0][:80] or "episode",
        group_id=GROUP_ID,
        labels=[],
        source=EpisodeType.text,
        source_description=description,
        content=content,
        valid_at=valid_at or now(),
        created_at=now(),
        entity_edges=[],
    )
    await ep.save(driver())
    return ep


async def save_edge(source: EntityNode, relation: str, target: EntityNode,
                    fact: str, valid_at: datetime | None = None,
                    attributes: dict[str, Any] | None = None,
                    episode_uuid: str | None = None,
                    embedding: list[float] | None = None) -> EntityEdge:
    """Store a fact while keeping its validity separate from its recording time.

    Args:
        source: The existing subject entity.
        relation: The relationship name.
        target: The existing object entity.
        fact: The complete fact sentence.
        valid_at: When the fact became true; None means unknown.
        attributes: Optional fact metadata.
        episode_uuid: Optional shared provenance record.
        embedding: Optional fact embedding.

    Returns:
        The saved edge, with created_at set to the recording time.
    """
    edge = EntityEdge(
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        group_id=GROUP_ID,
        name=relation,
        fact=fact,
        created_at=now(),
        valid_at=valid_at,
        attributes=attributes or {},
        episodes=[episode_uuid] if episode_uuid else [],
        fact_embedding=embedding,
    )
    await edge.save(driver())
    return edge


async def get_edge(fact_id: str) -> EntityEdge:
    """Read one fact without relying on FalkorDB's relationship UUID index.

    Args:
        fact_id: The complete fact ID returned by a save, search, or inventory.

    Returns:
        The uniquely matching fact in the configured memory group. Its vector
        is not loaded; metadata updates preserve the database's stored vector.

    Raises:
        EdgeNotFoundError: If the fact does not exist.
        ValueError: If the ID is ambiguous or belongs to another memory group.
    """
    # Indexed UUID equality can miss existing relationships that search still returns.
    records, _, _ = await driver().execute_query(
        "MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity) "
        "WHERE toString(e.uuid) = $uuid RETURN "
        + get_entity_edge_return_query(driver().provider) + " LIMIT 2",
        uuid=fact_id,
    )
    if not records:
        raise EdgeNotFoundError(fact_id)
    if len(records) != 1:
        raise ValueError("Multiple relationships share this ID; refusing an ambiguous lookup")
    edge = get_entity_edge_from_record(records[0], driver().provider)
    if edge.group_id != GROUP_ID:
        raise ValueError("Relationship does not belong to the configured memory group")
    return edge


async def update_edge(edge: EntityEdge) -> None:
    """Update an existing fact in place while retaining its stored embedding.

    Args:
        edge: The updated fact model. Its ID, group, and endpoints must still
            identify exactly one existing relationship. The model's embedding
            is ignored; the existing database vector is retained atomically.

    Returns:
        None after the update is confirmed. This operation never creates an edge.

    Raises:
        ValueError: If the model belongs to another memory group.
        RuntimeError: If the fact is missing, ambiguous, or its identity changed.
    """
    if edge.group_id != GROUP_ID:
        raise ValueError("Relationship does not belong to the configured memory group")
    properties = edge.model_dump(exclude={
        "source_node_uuid", "target_node_uuid", "attributes", "fact_embedding"})
    properties.update(source_uuid=edge.source_node_uuid, target_uuid=edge.target_node_uuid)
    reserved = set(type(edge).model_fields) | {"source_uuid", "target_uuid"}
    properties.update({key: value for key, value in (edge.attributes or {}).items()
                       if key not in reserved})
    records, _, _ = await driver().execute_query(
        "MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity) "
        "WHERE toString(e.uuid) = $uuid "
        "WITH collect(e) AS matches WHERE size(matches) = 1 "
        "UNWIND matches AS e "
        "WITH e WHERE e.group_id = $group_id "
        "AND toString(startNode(e).uuid) = $source_uuid "
        "AND toString(endNode(e).uuid) = $target_uuid "
        "WITH e, e.fact_embedding AS stored_embedding "
        "SET e = $properties SET e.fact_embedding = stored_embedding "
        "RETURN e.uuid AS uuid",
        uuid=edge.uuid, group_id=GROUP_ID, source_uuid=edge.source_node_uuid,
        target_uuid=edge.target_node_uuid, properties=properties,
    )
    if len(records) != 1:
        raise RuntimeError("Relationship no longer has one matching identity; retry the lookup")


async def link_episode(ep: EpisodicNode, edge_uuids: list[str]) -> None:
    """Point an episode back at the facts drawn from it.

    The link is stored both ways so `get_story` can go fact -> episode and
    "how did I come to know things about X" can go episode -> facts.
    """
    ep.entity_edges = list(dict.fromkeys(list(ep.entity_edges) + edge_uuids))
    await ep.save(driver())


async def edges_for_entity(node_uuid: str) -> list[EntityEdge]:
    """Every fact an entity takes part in, in either direction."""
    return await EntityEdge.get_by_node_uuid(driver(), node_uuid)


async def page_edges(limit: int, cursor: str | None = None) -> list[EntityEdge]:
    """Read a bounded page of facts from the configured memory group.

    Args:
        limit: Maximum number of records to read.
        cursor: Exclude IDs at or above this boundary in descending UUID order.

    Returns:
        Facts in descending UUID order, or an empty list at the end.
    """
    # Cast before comparing: FalkorDB's indexed string ranges can include UUIDs above the boundary.
    boundary = "AND toString(e.uuid) < $cursor " if cursor is not None else ""
    records, _, _ = await driver().execute_query(
        "MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity) "
        "WHERE e.group_id = $group_id " + boundary + "RETURN "
        + get_entity_edge_return_query(driver().provider)
        + " ORDER BY e.uuid DESC LIMIT $limit",
        group_id=GROUP_ID, cursor=cursor, limit=limit,
    )
    return [get_entity_edge_from_record(record, driver().provider) for record in records]


async def page_entities(limit: int, cursor: str | None = None) -> list[EntityNode]:
    """Read a bounded entity inventory in the configured memory group.

    Args:
        limit: Maximum entities to read, including any pagination lookahead.
        cursor: Exclude IDs at or above this boundary in descending UUID order.

    Returns:
        Entities in descending UUID order, including entities without relationships.
        Embedding vectors are not loaded. Concurrent writes are not a snapshot.
    """
    boundary = "AND toString(n.uuid) < $cursor " if cursor is not None else ""
    records, _, _ = await driver().execute_query(
        "MATCH (n:Entity) WHERE n.group_id = $group_id " + boundary + "RETURN "
        + get_entity_node_return_query(driver().provider)
        + " ORDER BY n.uuid DESC LIMIT $limit",
        group_id=GROUP_ID, cursor=cursor, limit=limit,
    )
    return [get_entity_node_from_record(record, driver().provider) for record in records]


async def adjacent_edges(node_uuids: list[str], excluded: list[str], limit: int,
                         direction: Direction = "both", include_superseded: bool = False,
                         preferred_uuid: str | None = None) -> list[EntityEdge]:
    """Read a bounded, eligible traversal frontier without crossing memory groups.

    Args:
        node_uuids: Entities whose adjacent facts should be read.
        excluded: Fact IDs already visited.
        limit: Maximum number of facts to return.
        direction: Follow facts in either, subject-to-object, or reverse direction.
        include_superseded: Allow historical facts to participate in the traversal.
        preferred_uuid: Prefer edges reaching this entity when limiting a frontier.

    Returns:
        Distinct facts, prioritizing the preferred entity then ordered by fact ID.

    Raises:
        ValueError: If direction is invalid.
        TimeoutError: If the database query exceeds two seconds.
    """
    frontiers = {
        "both": "(n.uuid IN $frontier OR m.uuid IN $frontier)",
        "outgoing": "n.uuid IN $frontier",
        "incoming": "m.uuid IN $frontier",
    }
    if direction not in frontiers:
        raise ValueError("direction must be both, outgoing, or incoming")
    if not node_uuids:
        return []
    validity = "" if include_superseded else "AND e.invalid_at IS NULL "
    # Indexed relationship-group equality can hide edges reachable through their entities.
    query = (
        "MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity) "
        "WHERE " + frontiers[direction] + " AND n.group_id = $group_id "
        "AND m.group_id = $group_id AND toString(e.group_id) = $group_id "
        "AND NOT e.uuid IN $excluded " + validity
        + "WITH e, CASE WHEN n.uuid = $preferred OR m.uuid = $preferred THEN 0 ELSE 1 END AS priority "
        "RETURN " + get_entity_edge_return_query(driver().provider)
        + " ORDER BY priority, e.uuid LIMIT $limit"
    )
    graph = driver().client.select_graph(GRAPH_DATABASE)
    try:
        result = await graph.ro_query(query, params={
            "frontier": node_uuids, "group_id": GROUP_ID, "excluded": excluded,
            "preferred": preferred_uuid, "limit": limit,
        }, timeout=2000)
    except ResponseError as error:
        if "timed out" in str(error).lower() or "timeout" in str(error).lower():
            raise TimeoutError("Graph traversal query timed out; narrow the exploration") from error
        raise
    fields = [column[1] for column in result.header]
    return [get_entity_edge_from_record(dict(zip(fields, row, strict=True)), driver().provider)
            for row in result.result_set]


async def vocabulary() -> dict[str, list[str]]:
    """Entity types and relation names already in use.

    Nothing in the store constrains these, so the risk is drift — `PREFERS`
    this month and `LIKES` the next, splitting one relation into two. Showing
    what exists is cheaper than policing it.

    Uses graphiti's own accessors rather than Cypher: the edge model differs by
    backend (FalkorDB stores a relationship, Kuzu an intermediate node), so
    hand-written queries silently return nothing when the driver changes.
    """
    nodes = await EntityNode.get_by_group_ids(driver(), [GROUP_ID])
    types = {label for n in nodes for label in n.labels if label != "Entity"}
    edges = await EntityEdge.get_by_group_ids(driver(), [GROUP_ID])
    return {"entity_types": sorted(types),
            "relations": sorted({e.name for e in edges if e.name})}
