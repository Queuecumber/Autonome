"""Bounded graph exploration using stored facts and NetworkX path algorithms."""

from dataclasses import dataclass

from graphiti_core.edges import EntityEdge
import networkx as nx

from graphiti_mcp import store

MAX_HOPS = 6
MAX_FACTS = 100
TIMEOUT_SECONDS = 10


@dataclass
class Exploration:
    """A bounded graph, its original facts, and whether eligible facts were omitted.

    Args:
        graph: Topology in the requested traversal direction, keyed by entity UUID.
        facts: Original directed relationships keyed by fact UUID.
        truncated: True when the fact budget left eligible facts unexplored.
    """
    graph: nx.MultiGraph | nx.MultiDiGraph
    facts: dict[str, EntityEdge]
    truncated: bool = False


def validate_limits(max_hops: int, limit: int, direction: str = "both") -> None:
    """Validate exploration settings before accessing the graph.

    Args:
        max_hops: Requested radius, from 1 to 6.
        limit: Maximum explored facts, from 1 to 100.
        direction: both, outgoing, or incoming.

    Raises:
        ValueError: If a setting is outside its supported bounds.
    """
    if not 1 <= max_hops <= MAX_HOPS:
        raise ValueError(f"max_hops must be between 1 and {MAX_HOPS}")
    if not 1 <= limit <= MAX_FACTS:
        raise ValueError(f"limit must be between 1 and {MAX_FACTS}")
    if direction not in {"both", "outgoing", "incoming"}:
        raise ValueError("direction must be both, outgoing, or incoming")


async def explore(root_uuid: str, max_hops: int, limit: int,
                  direction: store.Direction = "both", include_superseded: bool = False,
                  target_uuid: str | None = None) -> Exploration:
    """Expand eligible facts one layer at a time, within a fixed fact budget.

    Args:
        root_uuid: Starting entity in the configured memory group.
        max_hops: Maximum number of expansion layers, from 1 to 6.
        limit: Maximum facts retained in the exploration, from 1 to 100.
        direction: Traversal orientation; original fact directions are retained.
        include_superseded: Allow historical facts at every expansion step.
        target_uuid: Stop after reaching this entity, preferring it in each frontier.

    Returns:
        A bounded graph with original facts and an explicit truncation flag.
        At most one additional fact is read to detect truncation.

    Raises:
        ValueError: If exploration limits or direction are invalid.
        TimeoutError: If a database expansion query times out.
    """
    validate_limits(max_hops, limit, direction)
    topology = nx.MultiGraph() if direction == "both" else nx.MultiDiGraph()
    topology.add_node(root_uuid)
    result = Exploration(topology, {})
    if target_uuid == root_uuid:
        return result
    visited = {root_uuid}
    frontier = [root_uuid]
    for depth in range(max_hops):
        remaining = limit - len(result.facts)
        edges = await store.adjacent_edges(
            frontier, list(result.facts), remaining + 1, direction,
            include_superseded, preferred_uuid=target_uuid)
        result.truncated = len(edges) > remaining
        next_frontier = set()
        for edge in edges[:remaining]:
            result.facts[edge.uuid] = edge
            source, target = edge.source_node_uuid, edge.target_node_uuid
            if direction == "incoming":
                source, target = target, source
            topology.add_edge(source, target, key=edge.uuid)
            next_frontier.update({source, target} - visited)
        visited.update(next_frontier)
        if result.truncated or target_uuid in visited or not next_frontier:
            break
        frontier = sorted(next_frontier)
        if len(result.facts) == limit:
            if depth + 1 < max_hops:
                result.truncated = bool(await store.adjacent_edges(
                    frontier, list(result.facts), 1, direction, include_superseded))
            break
    return result


def shortest_route(exploration: Exploration, source: str, target: str,
                   max_hops: int) -> tuple[list[str], list[EntityEdge]] | None:
    """Find one shortest route in a bounded exploration.

    Args:
        exploration: The previously explored graph.
        source: Starting entity UUID.
        target: Destination entity UUID.
        max_hops: Maximum permitted path length.

    Returns:
        Ordered entity UUIDs and one original fact per step, or None when no
        route exists in the explored graph. Parallel facts are selected by UUID.
    """
    route = nx.single_source_shortest_path(exploration.graph, source, cutoff=max_hops).get(target)
    if route is None:
        return None
    facts = [exploration.facts[min(exploration.graph[left][right])]
             for left, right in zip(route, route[1:])]
    return route, facts


def distances(exploration: Exploration, root: str) -> dict[str, int]:
    """Return shortest hop counts from root within the bounded exploration."""
    return dict(nx.single_source_shortest_path_length(exploration.graph, root))
