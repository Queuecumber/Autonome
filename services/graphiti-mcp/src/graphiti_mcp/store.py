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
from typing import Any

from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode

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


async def upsert_entity(name: str, entity_type: str, summary: str = "",
                        attributes: dict[str, Any] | None = None,
                        embedding: list[float] | None = None) -> EntityNode:
    """Fetch an entity by name, or create it.

    An existing entity keeps its identity: a later mention adds the type and
    fills an empty summary rather than replacing what is already known.
    """
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
    edge = EntityEdge(
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        group_id=GROUP_ID,
        name=relation,
        fact=fact,
        created_at=now(),
        valid_at=valid_at or now(),
        attributes=attributes or {},
        episodes=[episode_uuid] if episode_uuid else [],
        fact_embedding=embedding,
    )
    await edge.save(driver())
    return edge


async def update_edge(edge: EntityEdge) -> None:
    """Persist a change to an existing edge without losing its vector.

    `get_by_uuid` does not populate `fact_embedding` — it has its own loader —
    so the obvious read-modify-write cycle saves a null over the embedding and
    quietly removes the fact from semantic search while leaving it in the
    graph. Every mutation has to come through here.
    """
    if edge.fact_embedding is None:
        try:
            await edge.load_fact_embedding(driver())
        except Exception:
            pass          # never had one; a write is still better than a loss
    await edge.save(driver())


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
