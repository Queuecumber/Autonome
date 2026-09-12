"""Knowledge-graph memory — facts with a timeline, and the stories behind them.

Durable facts and their narrative context live together here. Older markdown
memory remains useful as source material, but does not prescribe a separate
journal for new memory writes.

Every fact carries the sentence you wrote, so nothing is reduced to a schema;
the subject/relation/object triple is an index over that sentence, not a
replacement for it.
"""

import json
import logging
import os
from datetime import datetime
from uuid import UUID

from fastmcp import FastMCP
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.search.search_filters import (ComparisonOperator, DateFilter,
                                                 SearchFilters)
from graphiti_core.search.search_utils import (edge_fulltext_search,
                                                 edge_similarity_search)
from pydantic import BaseModel, Field

from graphiti_mcp import embed, store

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

mcp = FastMCP("graph", instructions=(
    """
# Knowledge Graph Memory

This graph is your durable long-term memory: facts describe people, events,
decisions, commitments, and relationships; their shared stories preserve what
happened and why it mattered. Keep useful narrative in `story` with the facts
it explains. Retrieve the story when a bare fact is not enough.

Existing markdown memories and Journal/ files are historical sources to consult
when needed. They do not prescribe a separate daily or heartbeat journal for new
memory writes. Follow this policy when older memory instructions conflict.

A fact is a sentence plus the two things it connects:

    "Max prefers function-level commit granularity"
     ^^^                ^^^^^^^^^^^^^^^^^^^^^^^^^
     Max  --PREFERS-->  commit granularity

Save the sentence you would have written anyway; naming the two entities and
the relation is what lets you find it again by either one.

Facts are never overwritten. When something stops being true, supersede it —
the old fact stays, with the date it stopped applying, so you can still answer
"what did I used to think?".

When several facts come out of one conversation, save them together in a single
call with one `story`. `story` is the narrative text; `source` is its attribution,
such as a message link or an old journal filename. Source alone is retained as
provenance with an empty story; it does not generate narrative. Use `get_story`
to inspect what was retained, and reuse its episode_id for later related facts.

Dates in a sentence do not set the timeline. For historical imports, explicitly
set each fact's `valid_at` to when it became true, not when you copied it. Use
explicit null when that date is unknown, keeping any approximate date in the
sentence or story. A batch date is only a fallback for facts that omit their
own date. `recorded_at` is the separate, automatic recording time. Correct an
imported date with `set_fact_valid_at`; use `supersede_fact` when reality changed.

Use `list_facts` for an audit when you do not know what to search for. Follow its
next_cursor for further bounded pages; it includes historical facts. Do not
dump the entire inventory into context on every turn. On a fresh session, look
up the people, active projects, decisions, and commitments relevant to the work.

Save information likely to matter after this context is gone. Search before
recording a repeated fact. Keep useful detail, but do not turn routine heartbeat
ticks, acknowledgements, or tool validation tests into durable facts. There is
no daily fact quota. Superseding retains history and does not reclaim storage;
review relevance and duplication rather than pruning solely because counts grew.
"""
))


class Fact(BaseModel):
    """One relationship, as a sentence plus the two things it connects."""
    subject: str = Field(description="The entity the fact is about, e.g. 'Max'.")
    subject_type: str = Field(default="", description=(
        "Its kind, e.g. 'Person' or 'Chat room'. Whitespace is normalized to underscores."))
    relation: str = Field(description="The relationship, e.g. 'PREFERS'.")
    object: str = Field(description="What it relates to, e.g. 'commit granularity'.")
    object_type: str = Field(default="", description=(
        "Its kind, e.g. 'Preference' or 'Character artifact'. Whitespace is normalized to underscores."))
    fact: str = Field(description="The full sentence, as you would write it.")
    valid_at: datetime | None = Field(default=None, description=(
        "When this fact became true. Set the historical timestamp when importing old facts; "
        "explicit null means unknown. Omit to inherit the batch date or the recording time. "
        "Dates without a timezone are interpreted as UTC."))


@mcp.tool
async def save_facts(facts: list[Fact], story: str = "", episode_id: str = "",
                     valid_at: datetime | None = None,
                     source: str = "") -> dict:
    """Record one or more facts, optionally with the story behind them.

    Args:
        facts: The relationships to record. Save everything that came out of
            one conversation in a single call so they share a story.
        story: The actual narrative in markdown: what happened and how you
            learned the facts. This is separate from the source attribution.
        episode_id: Attach to an existing story instead of writing a new one —
            use the `episode_id` from a search result when adding to something
            you already recorded. Do not combine with story or source.
        valid_at: Fallback date for facts that omit their own valid_at, and the
            shared source event's date. Defaults to now. Use per-fact dates for
            imports covering different events; explicit null on a fact keeps
            its date unknown. Dates without a timezone are interpreted as UTC.
        source: Attribution such as a message link or journal filename. Kept
            even without a story; it does not supply or generate narrative text.

    Returns:
        The stored `facts` (with their ids) and the `episode_id` they share.
        For a nonempty batch, story or source creates an episode; otherwise
        its ID is null. An empty batch is a no-op.

    Raises:
        ValueError: If episode_id is combined with story or source.
    """
    if episode_id and (story or source):
        raise ValueError("Use episode_id to reuse provenance, or story/source to create it")
    if not facts:
        return {"facts": [], "episode_id": episode_id or None}
    await store.ensure_indices()
    batch_date = store.utc_time(valid_at) if valid_at is not None else store.now()

    ep: EpisodicNode | None = None
    if episode_id:
        ep = await EpisodicNode.get_by_uuid(store.driver(), episode_id)
    elif story or source:
        ep = await store.save_episode(story, source, batch_date)

    saved, uuids = [], []
    for f in facts:
        fact_date = batch_date
        if "valid_at" in f.model_fields_set:
            fact_date = store.utc_time(f.valid_at) if f.valid_at is not None else None
        subj = await store.upsert_entity(
            f.subject, f.subject_type, embedding=await embed.embed(f.subject))
        obj = await store.upsert_entity(
            f.object, f.object_type, embedding=await embed.embed(f.object))
        edge = await store.save_edge(
            subj, f.relation, obj, f.fact, valid_at=fact_date,
            episode_uuid=ep.uuid if ep else None,
            embedding=await embed.embed(f.fact))
        uuids.append(edge.uuid)
        saved.append({"fact_id": edge.uuid, "fact": edge.fact,
                      "subject": subj.name, "relation": edge.name,
                      "object": obj.name,
                      "episode_id": ep.uuid if ep else None,
                      "recorded_at": edge.created_at.isoformat(),
                      "valid_at": edge.valid_at.isoformat() if edge.valid_at else None})

    if ep is not None:
        await store.link_episode(ep, uuids)
    logger.info("Saved %d fact(s)%s", len(saved), " with story" if ep else "")
    return {"facts": saved, "episode_id": ep.uuid if ep else None}


@mcp.tool
async def search_facts(query: str, limit: int = 10,
                       include_superseded: bool = False) -> list[dict]:
    """Find facts by meaning.

    Args:
        query: What you are looking for, in your own words.
        limit: How many matching facts to return, excluding superseded facts
            unless include_superseded is set.
        include_superseded: Also return facts that have stopped being true —
            use this when you want history rather than the current picture.

    Returns:
        Matching facts, each with its `fact_id`, validity dates, and the
        `episode_id` of the story behind it (pass that to `get_story`).
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

    return [await _render(edge) for edge in list(found.values())[:limit]]


@mcp.tool
async def list_facts(limit: int = 50, cursor: str | None = None) -> dict:
    """Audit stored facts without needing a search query.

    Args:
        limit: Page size, from 1 to 100.
        cursor: The previous page's next_cursor. Omit for the first page.

    Returns:
        A facts page and next_cursor, which is null at the end. Includes
        superseded facts for auditing. Pages use descending fact-ID order,
        not date or relevance order. Concurrent writes are not a snapshot;
        restart an audit to include facts saved while paging.

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
    return {
        "facts": [await _render(edge) for edge in page],
        "next_cursor": page[-1].uuid if len(edges) > limit else None,
    }


@mcp.tool
async def supersede_fact(fact_id: str, invalid_at: datetime | None = None,
                         reason: str = "") -> dict:
    """Mark a fact as no longer true, keeping it as history.

    Use this instead of deleting when something *changed* — the old fact stays
    queryable with the date it stopped applying. Deleting is for a fact you
    recorded wrongly, which is a different thing.

    Args:
        fact_id: From a search result.
        invalid_at: When it stopped being true. Defaults to now.
        reason: Optional note about what changed.
    """
    edge = await EntityEdge.get_by_uuid(store.driver(), fact_id)
    edge.invalid_at = invalid_at or store.now()
    if reason:
        edge.attributes = {**(edge.attributes or {}), "superseded_reason": reason}
    await store.update_edge(edge)
    return await _render(edge)


@mcp.tool
async def set_fact_valid_at(fact_id: str, valid_at: datetime | None,
                            reason: str = "") -> dict:
    """Correct a fact's historical date without replacing the fact.

    Args:
        fact_id: The ID from a fact search, inventory, or save response.
        valid_at: Correct time when the fact became true, or null if unknown.
            Dates without a timezone are interpreted as UTC. Keep uncertain
            dates in the sentence or story instead of guessing an exact date.
        reason: Why the previously recorded date was wrong.

    Returns:
        The fact with its corrected valid_at and correction history. Its ID,
        original recorded_at, sentence, story links, and embedding are retained.

    Raises:
        ValueError: If the fact belongs to another group or the new date is
            later than its invalid_at date.
        EdgeNotFoundError: If no fact has the supplied ID.
    """
    edge = await EntityEdge.get_by_uuid(store.driver(), fact_id)
    if edge.group_id != store.GROUP_ID:
        raise ValueError("Fact does not belong to the configured memory group")
    if valid_at is not None:
        valid_at = store.utc_time(valid_at)
        if edge.invalid_at is not None and valid_at > store.utc_time(edge.invalid_at):
            raise ValueError("valid_at cannot be later than invalid_at")
    if edge.valid_at == valid_at:
        return await _render(edge)
    corrections = json.loads((edge.attributes or {}).get("valid_at_corrections", "[]"))
    corrections.append({
        "previous_valid_at": edge.valid_at.isoformat() if edge.valid_at else None,
        "valid_at": valid_at.isoformat() if valid_at else None,
        "corrected_at": store.now().isoformat(),
        "reason": reason,
    })
    edge.valid_at = valid_at
    edge.attributes = {**(edge.attributes or {}), "valid_at_corrections": json.dumps(corrections)}
    await store.update_edge(edge)
    return await _render(edge)


@mcp.tool
async def get_story(episode_id: str) -> dict:
    """Read the story behind a fact — how you came to record it.

    Args:
        episode_id: From a search result or a `save_facts` response.

    Returns:
        The narrative in story and its attribution in source. Story is empty
        for source-only provenance. recorded_at is when the episode was saved;
        valid_at is when the described source event occurred.
    """
    ep = await EpisodicNode.get_by_uuid(store.driver(), episode_id)
    return {"episode_id": ep.uuid, "story": ep.content,
            "source": ep.source_description,
            "recorded_at": ep.created_at.isoformat(),
            "valid_at": ep.valid_at.isoformat() if ep.valid_at else None,
            "fact_count": len(ep.entity_edges or [])}


@mcp.tool
async def get_entity(name: str) -> dict:
    """Everything you know about one thing, and how it connects.

    Args:
        name: Exact name, as recorded.

    Returns:
        The entity and every fact it takes part in, in either direction.
        Types use canonical labels, e.g. 'Chat_room' for 'Chat room'.
    """
    node = await store.find_entity(name)
    if node is None:
        return {"found": False, "name": name}
    facts = [await _render(e) for e in await store.edges_for_entity(node.uuid)]
    return {"found": True, "name": node.name,
            "types": [l for l in node.labels if l != "Entity"],
            "summary": node.summary, "attributes": node.attributes or {},
            "facts": facts}


@mcp.tool
async def list_vocabulary() -> dict:
    """The entity types and relation names you have already used.

    Check this before inventing a new one — reusing `PREFERS` keeps facts
    findable together, where adding `LIKES` quietly splits them in two.
    Entity types use canonical labels, e.g. 'Chat_room' for 'Chat room'.
    """
    return await store.vocabulary()


async def _render(edge: EntityEdge) -> dict:
    src = await EntityNode.get_by_uuid(store.driver(), edge.source_node_uuid)
    tgt = await EntityNode.get_by_uuid(store.driver(), edge.target_node_uuid)
    result = {
        "fact_id": edge.uuid,
        "fact": edge.fact,
        "subject": src.name,
        "relation": edge.name,
        "object": tgt.name,
        "recorded_at": edge.created_at.isoformat(),
        "valid_at": edge.valid_at.isoformat() if edge.valid_at else None,
        "invalid_at": edge.invalid_at.isoformat() if edge.invalid_at else None,
        "superseded": edge.invalid_at is not None,
        "episode_id": (edge.episodes or [None])[0],
    }
    if corrections := (edge.attributes or {}).get("valid_at_corrections"):
        result["valid_at_corrections"] = json.loads(corrections)
    return result


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0",
            port=int(os.environ.get("GRAPH_MCP_PORT", "8005")))
