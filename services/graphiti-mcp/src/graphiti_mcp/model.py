"""Long-term memory: stories indexed by entities and directed relationships.

Descriptions retain what the agent recorded, including impressions, feelings,
and uncertainty. Graph structure helps locate a story; it does not establish
the truth of its contents. Storage retains Graphiti's existing property names.
"""

import asyncio
import logging
from datetime import datetime
from typing import Annotated, Literal, Self
from uuid import UUID

from graphiti_core.edges import EntityEdge
from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.search.search_filters import (ComparisonOperator, DateFilter,
                                                 SearchFilters)
from graphiti_core.search.search_utils import (edge_fulltext_search,
                                                 edge_similarity_search,
                                                 node_fulltext_search,
                                                 node_similarity_search)
from pydantic import BaseModel, Field, JsonValue, TypeAdapter, computed_field

from graphiti_mcp import embed, store, traversal

HopCount = Annotated[int, Field(ge=1, le=traversal.MAX_HOPS)]
RelationshipBudget = Annotated[int, Field(ge=1, le=traversal.MAX_RELATIONSHIPS)]

logger = logging.getLogger(__name__)


class DateCorrection(BaseModel):
    """A saved date adjustment: previous/new validity, recording time, and reason."""

    previous_valid_at: datetime | None = None
    valid_at: datetime | None
    corrected_at: datetime
    reason: str = ""


_corrections = TypeAdapter(list[DateCorrection])


class RelationshipRecord(BaseModel):
    """A persisted relationship with endpoints, provenance, validity, and typed audit history."""

    relationship_id: str
    description: str
    subject: str
    subject_id: str
    relation: str
    object: str
    object_id: str
    evidence_kind: Literal["reported", "inferred", "uncertain", "unspecified"] = "unspecified"
    rationale: str = ""
    recorded_at: datetime
    valid_at: datetime | None
    invalid_at: datetime | None
    story_id: str | None
    valid_at_corrections: list[DateCorrection] = Field(default_factory=list)

    @computed_field
    @property
    def superseded(self) -> bool:
        """Return whether this relationship has a recorded end to its validity."""
        return self.invalid_at is not None

    @computed_field
    @property
    def uri(self) -> str:
        """Return the MCP resource URI for this relationship."""
        return f"memory:///relationships/{self.relationship_id}"

    @computed_field
    @property
    def story_uri(self) -> str | None:
        """Return the full story's resource URI, or None without linked provenance."""
        return f"memory:///stories/{self.story_id}" if self.story_id is not None else None


class PathStep(RelationshipRecord):
    """A recorded relationship plus its traversal endpoints and walk direction."""

    from_entity_id: str
    to_entity_id: str
    traversed_forward: bool

    @classmethod
    def from_relationship(cls, relationship: RelationshipRecord, source: str, target: str) -> Self:
        """Attach source/target traversal IDs to a relationship without changing its direction.

        Args:
            relationship: The original directed relationship.
            source: Entity where the path enters this step.
            target: Entity where the path leaves this step.

        Returns:
            A typed path step retaining the relationship's history and provenance.
        """
        return cls(
            relationship_id=relationship.relationship_id, description=relationship.description,
            subject=relationship.subject, subject_id=relationship.subject_id,
            relation=relationship.relation, object=relationship.object, object_id=relationship.object_id,
            evidence_kind=relationship.evidence_kind, rationale=relationship.rationale,
            recorded_at=relationship.recorded_at, valid_at=relationship.valid_at,
            invalid_at=relationship.invalid_at, story_id=relationship.story_id,
            valid_at_corrections=relationship.valid_at_corrections,
            from_entity_id=source, to_entity_id=target,
            traversed_forward=relationship.subject_id == source,
        )


class Entity(BaseModel):
    """An entity's stable ID, name, and normalized types, without backend vectors."""

    entity_id: str
    name: str
    types: list[str]

    @computed_field
    @property
    def uri(self) -> str:
        """Return the MCP resource URI for this entity."""
        return f"memory:///entities/{self.entity_id}"

    @classmethod
    def from_node(cls, node: EntityNode) -> Self:
        """Coerce a Graphiti node into this public entity type, omitting its internal label.

        Args:
            node: The stored entity, already checked against the memory group.

        Returns:
            The entity ID, name, and canonical user-supplied types.
        """
        return cls(entity_id=node.uuid, name=node.name,
                   types=[label for label in node.labels if label != "Entity"])


class EntityMatch(Entity):
    """An entity search match with a bounded, explicitly truncated relationship preview."""

    summary: str = ""
    relationships: list[RelationshipRecord] = Field(default_factory=list)
    relationships_truncated: bool = False


class EntityVisit(Entity):
    """An explored entity with its distance in hops from the requested root."""

    hops: int = 0


class EntityResult(BaseModel):
    """An exact entity lookup; found=false retains the requested name and has no entity ID."""

    found: bool
    name: str
    entity_id: str | None = None
    types: list[str] = Field(default_factory=list)
    summary: str = ""
    attributes: dict[str, JsonValue] = Field(default_factory=dict)
    relationships: list[RelationshipRecord] = Field(default_factory=list)

    @computed_field
    @property
    def uri(self) -> str | None:
        """Return the entity resource URI, or None for a missing entity."""
        return f"memory:///entities/{self.entity_id}" if self.entity_id is not None else None


class Story(BaseModel):
    """A saved narrative and attribution, with explicit flags when used as a source preview."""

    story_id: str
    story: str
    source: str
    recorded_at: datetime
    valid_at: datetime | None
    relationship_count: int
    story_truncated: bool = False
    source_truncated: bool = False

    @computed_field
    @property
    def uri(self) -> str:
        """Return the MCP resource URI for the full story."""
        return f"memory:///stories/{self.story_id}"

    @classmethod
    def from_episode(cls, episode: EpisodicNode, story_chars: int | None = None,
                     source_chars: int | None = None) -> Self:
        """Coerce saved provenance into a story or bounded source preview.

        Args:
            episode: Stored provenance already checked against the memory group.
            story_chars: Maximum narrative characters, or None for the full story.
            source_chars: Maximum attribution characters, or None for all attribution.

        Returns:
            A story with native timestamps and explicit truncation indicators.
        """
        return cls(
            story_id=episode.uuid, story=episode.content[:story_chars],
            source=episode.source_description[:source_chars], recorded_at=episode.created_at,
            valid_at=episode.valid_at, relationship_count=len(episode.entity_edges or []),
            story_truncated=story_chars is not None and len(episode.content) > story_chars,
            source_truncated=source_chars is not None and len(episode.source_description) > source_chars,
        )


class MemorySaved(BaseModel):
    """Relationships recorded in one operation and the optional story they share."""

    relationships: list[RelationshipRecord]
    story_id: str | None

    @computed_field
    @property
    def story_uri(self) -> str | None:
        """Return the saved story's resource URI, or None for an unlinked batch."""
        return f"memory:///stories/{self.story_id}" if self.story_id is not None else None


class EntityPage(BaseModel):
    """A bounded entity inventory with a next-page cursor, or None at its end."""

    entities: list[Entity]
    next_cursor: str | None


class RelationshipPage(BaseModel):
    """A bounded relationship inventory with a next-page cursor, or None at its end."""

    relationships: list[RelationshipRecord]
    next_cursor: str | None


class Exploration(BaseModel):
    """A bounded neighborhood retaining original relationships and entity hop distances."""

    found: bool
    name: str
    max_hops: int
    direction: store.Direction
    entities: list[EntityVisit] = Field(default_factory=list)
    relationships: list[RelationshipRecord] = Field(default_factory=list)
    truncated: bool = False


class Path(BaseModel):
    """A bounded path lookup, with original relationships and explicit traversal directions."""

    found: bool
    source: str
    target: str
    max_hops: int
    directed: bool
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[PathStep] = Field(default_factory=list)
    hops: int | None = None
    explored_relationships: int = 0
    truncated: bool = False
    missing_entities: list[str] = Field(default_factory=list)


class RelationshipExplanation(BaseModel):
    """Saved reasoning and sources for one relationship; missing records have found=false."""

    found: bool
    relationship_id: str
    relationship: RelationshipRecord | None = None
    sources: list[Story] = Field(default_factory=list)
    missing_story_ids: list[str] = Field(default_factory=list)
    sources_truncated: bool = False
    superseded_reason: str = ""


class EntityExplanation(BaseModel):
    """Bounded explanations for an entity, with explicit absence and truncation states."""

    found: bool
    name: str
    entity_id: str | None = None
    types: list[str] = Field(default_factory=list)
    summary: str = ""
    relationships: list[RelationshipExplanation] = Field(default_factory=list)
    truncated: bool = False


class Vocabulary(BaseModel):
    """Entity types and relationship names already used in this memory group."""

    entity_types: list[str]
    relations: list[str]


class Relationship(BaseModel):
    """One relationship, as a sentence plus the two things it connects."""
    subject: str = Field(description="The subject entity's name, e.g. 'Max'.")
    subject_type: str = Field(default="", description=(
        "Its kind, e.g. 'Person' or 'Chat room'. Whitespace is normalized to underscores."))
    relation: str = Field(description="The relationship, e.g. 'PREFERS'.")
    object: str = Field(description="The object entity's name, e.g. 'commit granularity'.")
    object_type: str = Field(default="", description=(
        "Its kind, e.g. 'Preference' or 'Character artifact'. Whitespace is normalized to underscores."))
    description: str = Field(description=(
        "The relationship in your own words, including any feelings, impressions, or uncertainty."))
    valid_at: datetime | None = Field(default=None, description=(
        "When this relationship began to apply. Set the historical timestamp when importing memory; "
        "explicit null means unknown. Omit to inherit the batch date or the recording time. "
        "Dates without a timezone are interpreted as UTC."))
    evidence_kind: Literal["reported", "inferred", "uncertain", "unspecified"] = Field(
        default="unspecified", description=(
            "How you came to record this: a source statement, your interpretation, something "
            "uncertain, or unspecified. This does not require or imply objective verification."))
    rationale: str = Field(default="", description="Saved reasoning, context, or qualifications.")


async def save_memory(relationships: list[Relationship], story: str = "", story_id: str = "",
                     valid_at: datetime | None = None,
                     source: str = "") -> MemorySaved:
    """Record a memory, with the story behind it.

    Args:
        relationships: The relationships to record. Save everything that came out of
            one conversation in a single call so they share a story.
        story: The narrative to remember: an event, impression, feeling, note,
            summary, or verbatim text. This is separate from source attribution.
        story_id: Attach to an existing story instead of writing a new one —
            use the `story_id` from a search result when adding to something
            you already recorded. Do not combine with story or source.
        valid_at: Fallback date for relationships that omit their own valid_at, and the
            shared source event's date. Defaults to now. Use per-relationship dates for
            imports covering different events; explicit null on a relationship keeps
            its date unknown. Dates without a timezone are interpreted as UTC.
        source: Attribution such as a message link or journal filename. Kept
            even without a story; it does not supply or generate narrative text.

    Returns:
        The stored `relationships` (with their ids) and the `story_id` they share.
        For a nonempty batch, story or source creates a story record; otherwise
        its ID is null. An empty batch is a no-op.
        All supplied labels are validated before any write, so a rejected label
        leaves no partial relationships, entity changes, or provenance. Database failures
        during writing can still leave partial results; inspect memory before
        retrying a failed database operation. This is not a multi-write transaction.

    Raises:
        ValueError: If story_id is combined with story or source, or an entity
            type is invalid after whitespace normalization, or the story belongs
            to another memory group.
    """
    if story_id and (story or source):
        raise ValueError("Use story_id to reuse provenance, or story/source to create it")
    if not relationships:
        return MemorySaved(relationships=[], story_id=story_id or None)
    entity_types = [(store.normalize_entity_type(f.subject_type),
                     store.normalize_entity_type(f.object_type)) for f in relationships]
    await store.ensure_indices()
    batch_date = store.utc_time(valid_at) if valid_at is not None else store.now()

    ep: EpisodicNode | None = None
    if story_id:
        ep = await _story_node(story_id)
    elif story or source:
        ep = await store.save_episode(story, source, batch_date)

    saved, uuids = [], []
    for f, (subject_type, object_type) in zip(relationships, entity_types, strict=True):
        relationship_date = batch_date
        if "valid_at" in f.model_fields_set:
            relationship_date = store.utc_time(f.valid_at) if f.valid_at is not None else None
        subj = await store.upsert_entity(
            f.subject, subject_type, embedding=await embed.embed(f.subject))
        obj = await store.upsert_entity(
            f.object, object_type, embedding=await embed.embed(f.object))
        edge = await store.save_edge(
            subj, f.relation, obj, f.description, valid_at=relationship_date,
            attributes={"evidence_kind": f.evidence_kind, "rationale": f.rationale},
            episode_uuid=ep.uuid if ep else None,
            embedding=await embed.embed(f.description))
        uuids.append(edge.uuid)
        saved.append(await _relationship_record(edge, {subj.uuid: subj, obj.uuid: obj}))

    if ep is not None:
        await store.link_episode(ep, uuids)
    logger.info("Saved %d relationship(s)%s", len(saved), " with story" if ep else "")
    return MemorySaved(relationships=saved, story_id=ep.uuid if ep else None)


async def search(query: str, limit: RelationshipBudget = 10,
                 include_superseded: bool = False,
                 kind: Literal["entities", "relationships"] = "entities",
                 relationship_limit: Annotated[int, Field(ge=1, le=20)] = 5) -> list[EntityMatch] | list[RelationshipRecord]:
    """Find entities by name or meaning, or search relationship descriptions.

    Args:
        query: What you are looking for, in your own words.
        limit: Maximum matching entities or relationships, from 1 to 100.
        include_superseded: Include historical relationships. Entities themselves
            are not superseded and remain discoverable without current relationships.
        kind: entities searches entity names and summaries; relationships searches
            the descriptions recorded with stories.
        relationship_limit: Maximum attached relationships per entity, from 1 to 20.
            Only used for entity results; follow get_entity or explore for more.

    Returns:
        For entities, summaries with entity_id, relationships, and an explicit
        relationships_truncated flag. For relationships, descriptions with
        relationship_id, dates, and story_id for get_story. Keyword retrieval
        still works when embeddings are unavailable. Results are bounded,
        not an exhaustive inventory; search failures are logged.

    Raises:
        ValueError: If kind or a result limit is invalid.
        TimeoutError: If an entity's relationship-preview query exceeds its deadline.
    """
    if not 1 <= limit <= 100 or not 1 <= relationship_limit <= 20:
        raise ValueError("limit must be 1..100 and relationship_limit must be 1..20")
    if kind not in {"entities", "relationships"}:
        raise ValueError("kind must be entities or relationships")
    if kind == "entities":
        return await _search_entities(query, limit, include_superseded, relationship_limit)
    return await _search_relationships(query, limit, include_superseded)


async def _search_relationships(query: str, limit: int,
                                include_superseded: bool) -> list[RelationshipRecord]:
    """Search recorded descriptions using independent semantic and keyword retrieval.

    Args:
        query: Text to match against relationship descriptions.
        limit: Maximum matches to render after deduplication.
        include_superseded: Include historical relationships in each search mode.

    Returns:
        Rendered relationships in the configured group. A failed search mode is
        logged and the other mode remains available; neither failure invents data.
    """
    await store.ensure_indices()
    filters = SearchFilters()
    if not include_superseded:
        filters.invalid_at = [[DateFilter(comparison_operator=ComparisonOperator.is_null)]]

    # Two ways of being relevant, and they fail differently: vectors miss an
    # exact name they never saw, keywords miss a paraphrase. Run both and
    # merge — a name lookup should not depend on the embedder being reachable.
    found: dict[str, EntityEdge] = {}
    vector = await embed.embed(query, is_query=True)
    if vector is not None:
        try:
            for e in await edge_similarity_search(
                    store.driver(), vector, None, None, filters,
                    [store.GROUP_ID], limit, min_score=embed.MIN_SCORE):
                found[e.uuid] = e
        except Exception as e:
            logger.warning("Semantic search unavailable: %r", e)
    try:
        for e in await edge_fulltext_search(
                store.driver(), query, filters, [store.GROUP_ID], limit):
            found.setdefault(e.uuid, e)
    except Exception as e:
        logger.warning("Keyword search unavailable: %r", e)

    return [await _relationship_record(edge) for edge in list(found.values())[:limit]]


async def _search_entities(query: str, limit: int, include_superseded: bool,
                           relationship_limit: int) -> list[EntityMatch]:
    """Search entity names and summaries, with bounded relationship previews.

    Args:
        query: Entity name or description to retrieve.
        limit: Maximum distinct entities, prioritizing an exact-name match.
        include_superseded: Include historical relationships in previews.
        relationship_limit: Maximum relationships per entity preview.

    Returns:
        Entity summaries and relationship previews with truncation flags.
        Optional search-mode failures are logged without suppressing other modes.

    Raises:
        TimeoutError: If a relationship preview query exceeds its read deadline.
    """
    await store.ensure_indices()
    found: dict[str, EntityNode] = {}
    exact = await store.find_entity(query)
    if exact is not None:
        found[exact.uuid] = exact
    vector = await embed.embed(query, is_query=True)
    if vector is not None:
        try:
            for node in await node_similarity_search(
                    store.driver(), vector, SearchFilters(), [store.GROUP_ID], limit,
                    min_score=embed.MIN_SCORE):
                found.setdefault(node.uuid, node)
        except Exception as error:
            logger.warning("Entity semantic search unavailable: %r", error)
    try:
        for node in await node_fulltext_search(
                store.driver(), query, SearchFilters(), [store.GROUP_ID], limit):
            found.setdefault(node.uuid, node)
    except Exception as error:
        logger.warning("Entity keyword search unavailable: %r", error)
    results = []
    for node in list(found.values())[:limit]:
        edges = await store.adjacent_edges(
            [node.uuid], [], relationship_limit + 1, include_superseded=include_superseded)
        result = EntityMatch.from_node(node)
        result.summary = node.summary
        result.relationships = [await _relationship_record(edge) for edge in edges[:relationship_limit]]
        result.relationships_truncated = len(edges) > relationship_limit
        results.append(result)
    return results


async def list_entities(limit: RelationshipBudget = 50, cursor: str | None = None) -> EntityPage:
    """Discover saved entities without a search query, including isolated entities.

    Args:
        limit: Page size, from 1 to 100.
        cursor: The previous next_cursor, or omit for the first page.

    Returns:
        entities with entity_id, name, and types, plus next_cursor (null at the
        end). Pages are in descending entity-ID order, not date or relevance
        order. Concurrent writes are not a snapshot. Use get_entity or explore
        for relationships and their story links.

    Raises:
        ValueError: If limit is invalid or cursor is not a UUID.
    """
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    if cursor is not None:
        cursor = str(UUID(cursor))
    await store.ensure_indices()
    nodes = await store.page_entities(limit + 1, cursor)
    page = nodes[:limit]
    return EntityPage(entities=[Entity.from_node(node) for node in page],
                      next_cursor=page[-1].uuid if len(nodes) > limit else None)


async def list_relationships(limit: int = 50, cursor: str | None = None) -> RelationshipPage:
    """Audit stored relationships without needing a search query.

    Args:
        limit: Page size, from 1 to 100.
        cursor: The previous page's next_cursor. Omit for the first page.

    Returns:
        A relationships page and next_cursor, which is null at the end. Includes
        superseded relationships for auditing. Pages use descending relationship-ID order,
        not date or relevance order. Concurrent writes are not a snapshot;
        restart an audit to include relationships saved while paging.

    Raises:
        ValueError: If limit is outside its bounds or cursor is not a UUID.
    """
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    if cursor is not None:
        cursor = str(UUID(cursor))
    await store.ensure_indices()
    edges = await store.page_edges(limit + 1, cursor)
    page = edges[:limit]
    return RelationshipPage(relationships=[await _relationship_record(edge) for edge in page],
                            next_cursor=page[-1].uuid if len(edges) > limit else None)


async def explore(name: str, max_hops: HopCount = 2, limit: RelationshipBudget = 50,
                           direction: store.Direction = "both",
                           include_superseded: bool = False) -> Exploration:
    """Explore relationships around an entity without an embedding request.

    Args:
        name: Exact entity name, as recorded.
        max_hops: Expansion depth, from 1 to 6.
        limit: Maximum explored relationships, from 1 to 100; one extra may detect truncation.
        direction: both, outgoing (subject to object), or incoming (reverse).
        include_superseded: Permit historical relationships at every hop. By default
            they cannot appear in results or act as hidden traversal bridges.

    Returns:
        found, entities with hop distances, original directed relationships, and truncated.
        A truncated graph is incomplete. Entities on the radius boundary are not
        expanded further. Concurrent writes are not a snapshot.

    Raises:
        ValueError: If limits or direction are invalid.
        TimeoutError: If the ten-second operation or a two-second query times out.
        RuntimeError: If entities change or disappear while assembling the result.
    """
    traversal.validate_limits(max_hops, limit, direction)
    async with asyncio.timeout(traversal.TIMEOUT_SECONDS):
        root = await store.find_entity(name)
        if root is None:
            return Exploration(found=False, name=name, max_hops=max_hops, direction=direction)
        explored = await traversal.explore(root.uuid, max_hops, limit, direction, include_superseded)
        nodes = await _load_nodes(list(explored.graph))
        hops = traversal.distances(explored, root.uuid)
        ordered = sorted(nodes.values(), key=lambda node: (hops[node.uuid], node.name, node.uuid))
        entities = []
        for node in ordered:
            entity = EntityVisit.from_node(node)
            entity.hops = hops[node.uuid]
            entities.append(entity)
        return Exploration(
            found=True, name=name, max_hops=max_hops, direction=direction, entities=entities,
            relationships=[await _relationship_record(edge, nodes) for edge in explored.relationships.values()],
            truncated=explored.truncated,
        )


async def find_path(source: str, target: str, max_hops: HopCount = 4,
                    limit: RelationshipBudget = 100, directed: bool = False,
                    include_superseded: bool = False) -> Path:
    """Find one shortest connection in a bounded exploration of recorded relationships.

    Args:
        source: Exact starting entity name.
        target: Exact destination entity name.
        max_hops: Maximum path length, from 1 to 6.
        limit: Maximum explored relationships, from 1 to 100, not the number of returned paths.
        directed: Follow subject-to-object direction only. Otherwise either direction is allowed.
        include_superseded: Allow historical relationships. Their dates may not overlap,
            so a historical chain need not have existed at one point in time.

    Returns:
        found, ordered entities and relationships, hops, explored_relationships, and truncated.
        traversed_forward on each step distinguishes its walk direction from
        the original relationship direction. If truncated, the search is incomplete;
        found=false means only that this bounded exploration found no route.
        A connection is not an assertion of a new transitive or causal relationship.

    Raises:
        ValueError: If limits are invalid.
        TimeoutError: If the ten-second operation or a two-second query times out.
        RuntimeError: If entities change or disappear while assembling the result.
    """
    traversal.validate_limits(max_hops, limit)
    async with asyncio.timeout(traversal.TIMEOUT_SECONDS):
        start = await store.find_entity(source)
        end = await store.find_entity(target)
        if start is None or end is None:
            return Path(found=False, source=source, target=target,
                        max_hops=max_hops, directed=directed,
                        missing_entities=list(dict.fromkeys(
                            name for name, node in [(source, start), (target, end)] if node is None)))
        explored = await traversal.explore(
            start.uuid, max_hops, limit, "outgoing" if directed else "both",
            include_superseded, target_uuid=end.uuid)
        route = traversal.shortest_route(explored, start.uuid, end.uuid, max_hops)
        result = Path(found=route is not None, source=source, target=target,
                      max_hops=max_hops, directed=directed,
                      explored_relationships=len(explored.relationships), truncated=explored.truncated)
        if route is not None:
            node_ids, edges = route
            nodes = await _load_nodes(node_ids)
            result.entities = [Entity.from_node(nodes[node_id]) for node_id in node_ids]
            result.relationships = [
                PathStep.from_relationship(await _relationship_record(edge, nodes), left, right)
                for left, right, edge in zip(node_ids, node_ids[1:], edges)
            ]
            result.hops = len(edges)
        return result


async def explain_relationship(relationship_id: str,
                       source_limit: Annotated[int, Field(ge=1, le=10)] = 5,
                       story_chars: Annotated[int, Field(ge=0, le=5000)] = 2000) -> RelationshipExplanation:
    """Inspect the stored basis for a claim without inventing an explanation.

    Args:
        relationship_id: ID from a search, traversal, inventory, or save result.
        source_limit: Maximum linked provenance records to return, from 1 to 10.
        story_chars: Maximum narrative characters per source, from 0 to 5000.
            Use get_story for a full narrative after inspecting the explanation.

    Returns:
        found, the relationship with its evidence label and rationale, source records,
        missing_story_ids, and sources_truncated. Legacy relationships have unspecified
        evidence, not a guessed confidence. Source attributions are capped at
        1000 characters; each source reports its own truncation flags. Date
        corrections and the supersession reason are retained in the explanation.

    Raises:
        ValueError: If the ID or limits are invalid, or the relationship belongs to another group.
        TimeoutError: If the ten-second read budget is exceeded.
        RuntimeError: If referenced entities are missing or outside the memory group.
    """
    relationship_id = str(UUID(relationship_id))
    if not 1 <= source_limit <= 10 or not 0 <= story_chars <= 5000:
        raise ValueError("source_limit must be 1..10 and story_chars must be 0..5000")
    async with asyncio.timeout(traversal.TIMEOUT_SECONDS):
        try:
            edge = await store.get_edge(relationship_id)
        except EdgeNotFoundError:
            return RelationshipExplanation(found=False, relationship_id=relationship_id)
        nodes = await _load_nodes([edge.source_node_uuid, edge.target_node_uuid])
        story_ids = list(dict.fromkeys(edge.episodes or []))
        selected = story_ids[:source_limit]
        episodes = {episode.uuid: episode for episode in await EpisodicNode.get_by_uuids(
            store.driver(), selected) if episode.group_id == store.GROUP_ID}
        sources = []
        for story_id in selected:
            if story_id not in episodes:
                continue
            episode = episodes[story_id]
            sources.append(Story.from_episode(episode, story_chars=story_chars, source_chars=1000))
        return RelationshipExplanation(
            found=True, relationship_id=edge.uuid, relationship=await _relationship_record(edge, nodes),
            sources=sources, missing_story_ids=[story_id for story_id in selected if story_id not in episodes],
            sources_truncated=len(story_ids) > source_limit,
            superseded_reason=(edge.attributes or {}).get("superseded_reason", ""),
        )


async def explain_entity(name: str, limit: Annotated[int, Field(ge=1, le=20)] = 5,
                         source_limit: Annotated[int, Field(ge=1, le=10)] = 3,
                         story_chars: Annotated[int, Field(ge=0, le=5000)] = 1000,
                         include_superseded: bool = False) -> EntityExplanation:
    """Inspect the stories, reasoning, and dates behind an entity's relationships.

    Args:
        name: Exact entity name, as recorded.
        limit: Maximum relationships to explain, from 1 to 20, ordered by ID.
        source_limit: Maximum source records per relationship, from 1 to 10.
        story_chars: Maximum narrative characters per source, from 0 to 5000.
        include_superseded: Include relationships that no longer apply.

    Returns:
        found, the entity summary, explained relationships, and truncated. Each
        explanation retains source truncation and missing-source indicators.
        Use explain_relationship for a specific relationship or get_story for
        an unabridged narrative. No new reasoning or confidence is generated.

    Raises:
        ValueError: If limits are invalid.
        TimeoutError: If the ten-second read budget is exceeded.
        RuntimeError: If referenced entities disappear or cross memory groups.
    """
    if not 1 <= limit <= 20 or not 1 <= source_limit <= 10 or not 0 <= story_chars <= 5000:
        raise ValueError("limit must be 1..20, source_limit 1..10, and story_chars 0..5000")
    async with asyncio.timeout(traversal.TIMEOUT_SECONDS):
        node = await store.find_entity(name)
        if node is None:
            return EntityExplanation(found=False, name=name)
        edges = await store.adjacent_edges(
            [node.uuid], [], limit + 1, include_superseded=include_superseded)
        entity = Entity.from_node(node)
        return EntityExplanation(
            found=True, entity_id=entity.entity_id, name=entity.name, types=entity.types,
            summary=node.summary,
            relationships=[await explain_relationship(edge.uuid, source_limit, story_chars)
                           for edge in edges[:limit]], truncated=len(edges) > limit,
        )


async def supersede(relationship_id: str, invalid_at: datetime | None = None,
                         reason: str = "") -> RelationshipRecord:
    """Mark a relationship as no longer applicable, keeping it as history.

    Use this when the world or your interpretation changes. Record the updated
    relationship with save_memory; the earlier description and story remain.

    Args:
        relationship_id: From a search result.
        invalid_at: When the relationship stopped applying. Defaults to now.
        reason: Optional note about what changed.

    Returns:
        The existing relationship marked as superseded, retaining its ID and vector.

    Raises:
        EdgeNotFoundError: If the relationship does not exist.
        ValueError: If the ID is ambiguous or belongs to another memory group.
        RuntimeError: If the relationship's identity changes before the update completes.
    """
    edge = await store.get_edge(relationship_id)
    edge.invalid_at = invalid_at or store.now()
    if reason:
        edge.attributes = {**(edge.attributes or {}), "superseded_reason": reason}
    await store.update_edge(edge)
    return await _relationship_record(edge)


async def set_valid_at(relationship_id: str, valid_at: datetime | None,
                            reason: str = "") -> RelationshipRecord:
    """Correct a relationship's historical date without replacing the relationship.

    Args:
        relationship_id: The ID from a relationship search, inventory, or save response.
        valid_at: Correct time when the relationship began to apply, or null if unknown.
            Dates without a timezone are interpreted as UTC. Keep uncertain
            dates in the sentence or story instead of guessing an exact date.
        reason: Why the previously recorded date was wrong.

    Returns:
        The relationship with its corrected valid_at and correction history. Its ID,
        original recorded_at, description, story links, and embedding are retained.

    Raises:
        ValueError: If the ID is ambiguous, the relationship belongs to another group,
            or the new date is later than its invalid_at date.
        EdgeNotFoundError: If no relationship has the supplied ID.
        RuntimeError: If the relationship's identity changes before the update completes.
    """
    edge = await store.get_edge(relationship_id)
    if valid_at is not None:
        valid_at = store.utc_time(valid_at)
        if edge.invalid_at is not None and valid_at > store.utc_time(edge.invalid_at):
            raise ValueError("valid_at cannot be later than invalid_at")
    if edge.valid_at == valid_at:
        return await _relationship_record(edge)
    corrections = _corrections.validate_json((edge.attributes or {}).get("valid_at_corrections", "[]"))
    corrections.append(DateCorrection(
        previous_valid_at=edge.valid_at, valid_at=valid_at,
        corrected_at=store.now(), reason=reason,
    ))
    edge.valid_at = valid_at
    edge.attributes = {**(edge.attributes or {}),
                       "valid_at_corrections": _corrections.dump_json(corrections).decode()}
    await store.update_edge(edge)
    return await _relationship_record(edge)


async def get_story(story_id: str) -> Story:
    """Read a saved memory's full narrative and attribution.

    Args:
        story_id: From a relationship, source record, or save_memory response.

    Returns:
        The narrative in story and its attribution in source. Story is empty
        for source-only provenance. recorded_at is when the story was saved;
        valid_at is when the described source event occurred.

    Raises:
        NodeNotFoundError: If the story does not exist.
        ValueError: If the ID is malformed or the story belongs to another group.
    """
    return Story.from_episode(await _story_node(story_id))


async def get_entity(name: str) -> EntityResult:
    """Everything you know about one thing, and how it connects.

    Args:
        name: Exact name, as recorded.

    Returns:
        The entity and every relationship it takes part in, including history.
        Types use canonical labels, e.g. 'Chat_room' for 'Chat room'.
    """
    node = await store.find_entity(name)
    if node is None:
        return EntityResult(found=False, name=name)
    return await _entity_result(node)


async def read_entity(entity_id: str) -> EntityResult:
    """Read an entity resource by its stable identifier, including relationship history.

    Args:
        entity_id: The complete entity UUID from a tool result or resource URI.

    Returns:
        The entity and its relationships in the configured memory group.

    Raises:
        NodeNotFoundError: If the entity does not exist.
        ValueError: If the ID is malformed or belongs to another group.
        RuntimeError: If a referenced entity is missing or outside the memory group.
    """
    node = await EntityNode.get_by_uuid(store.driver(), str(UUID(entity_id)))
    if node.group_id != store.GROUP_ID:
        raise ValueError("Entity does not belong to the configured memory group")
    return await _entity_result(node)


async def read_relationship(relationship_id: str) -> RelationshipRecord:
    """Read one relationship resource without relying on a search or traversal.

    Args:
        relationship_id: The complete relationship UUID from a result or resource URI.

    Returns:
        The stored relationship with typed dates, history, and provenance.

    Raises:
        EdgeNotFoundError: If the relationship does not exist.
        ValueError: If the ID is malformed, ambiguous, or belongs to another group.
        RuntimeError: If a referenced entity is missing or outside the memory group.
    """
    return await _relationship_record(await store.get_edge(str(UUID(relationship_id))))


async def _story_node(story_id: str) -> EpisodicNode:
    """Load a story by UUID, raising ValueError for malformed or foreign-group IDs.

    Args:
        story_id: The stored story UUID.

    Returns:
        The scoped backend provenance node, suitable for reading or linking.

    Raises:
        NodeNotFoundError: If no story has this ID.
        ValueError: If the ID is malformed or belongs to another memory group.
    """
    node = await EpisodicNode.get_by_uuid(store.driver(), str(UUID(story_id)))
    if node.group_id != store.GROUP_ID:
        raise ValueError("Story does not belong to the configured memory group")
    return node


async def _entity_result(node: EntityNode) -> EntityResult:
    """Coerce a scoped node and its stored relationships to the public entity result.

    Args:
        node: An entity already checked against the memory group.

    Returns:
        The entity with typed relationships, summary, and free-form JSON attributes.

    Raises:
        RuntimeError: If relationship endpoints disappear or cross memory groups.
    """
    entity = Entity.from_node(node)
    return EntityResult(
        found=True, name=entity.name, entity_id=entity.entity_id, types=entity.types,
        summary=node.summary, attributes=node.attributes or {},
        relationships=[await _relationship_record(edge) for edge in await store.edges_for_entity(node.uuid)],
    )


async def list_vocabulary() -> Vocabulary:
    """The entity types and relation names you have already used.

    Check this before inventing a new one — reusing `PREFERS` keeps relationships
    findable together, where adding `LIKES` quietly splits them in two.
    Entity types use canonical labels, e.g. 'Chat_room' for 'Chat room'.
    """
    return Vocabulary.model_validate(await store.vocabulary())


async def _load_nodes(node_ids: list[str]) -> dict[str, EntityNode]:
    """Load scoped entities for a result, raising RuntimeError if the graph changed."""
    nodes = {node.uuid: node for node in await EntityNode.get_by_uuids(store.driver(), node_ids)
             if node.group_id == store.GROUP_ID}
    if set(nodes) != set(node_ids):
        raise RuntimeError("Graph changed during exploration; retry the read")
    return nodes


async def _relationship_record(edge: EntityEdge, nodes: dict[str, EntityNode] | None = None) -> RelationshipRecord:
    """Coerce Graphiti storage into a relationship without exposing backend fields.

    Args:
        edge: The stored relationship in the configured group.
        nodes: Optional preloaded and scoped entity map for its endpoints.

    Returns:
        A relationship with native dates and typed correction history. Embedding
        vectors and backend property names do not enter the public data model.

    Raises:
        RuntimeError: If referenced entities are missing or outside the memory group.
        ValueError: If the relationship belongs to another memory group or its
            stored correction history cannot be parsed.
    """
    if edge.group_id != store.GROUP_ID:
        raise ValueError("Relationship does not belong to the configured memory group")
    if nodes is None:
        nodes = await _load_nodes([edge.source_node_uuid, edge.target_node_uuid])
    src, tgt = nodes[edge.source_node_uuid], nodes[edge.target_node_uuid]
    attributes = edge.attributes or {}
    return RelationshipRecord(
        relationship_id=edge.uuid, description=edge.fact,
        subject=src.name, subject_id=src.uuid, relation=edge.name,
        object=tgt.name, object_id=tgt.uuid,
        evidence_kind=attributes.get("evidence_kind", "unspecified"),
        rationale=attributes.get("rationale", ""), recorded_at=edge.created_at,
        valid_at=edge.valid_at, invalid_at=edge.invalid_at, story_id=(edge.episodes or [None])[0],
        valid_at_corrections=_corrections.validate_json(attributes.get("valid_at_corrections", "[]")),
    )
