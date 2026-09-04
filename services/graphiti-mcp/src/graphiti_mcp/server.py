"""Knowledge-graph memory — facts with a timeline, and the stories behind them.

Companion to the markdown memory service, not a replacement. The distinction
is whether a statement needs a *timeline* and *relationships*: "Max prefers
function-level commits" is about someone, could stop being true, and is worth
finding later from either end — that belongs here. A reflection on how the day
went belongs in the daily journal.

Every fact carries the sentence you wrote, so nothing is reduced to a schema;
the subject/relation/object triple is an index over that sentence, not a
replacement for it.
"""

import logging
import os
from datetime import datetime

from fastmcp import FastMCP
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.search.search_filters import SearchFilters
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

Structured long-term memory for things that are *about someone or something*
and that could change over time. Your daily markdown memory is still the place
for narrative and reflection; this is for facts you will want to look up later
from either end of a relationship.

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
call with one `story`. The story is the markdown of how you came to know these
things; it is kept once and shared by all of them, and you can pull it back
later when a bare fact is not enough.
"""
))


class Fact(BaseModel):
    """One relationship, as a sentence plus the two things it connects."""
    subject: str = Field(description="The entity the fact is about, e.g. 'Max'.")
    subject_type: str = Field(default="", description="Its kind, e.g. 'Person'.")
    relation: str = Field(description="The relationship, e.g. 'PREFERS'.")
    object: str = Field(description="What it relates to, e.g. 'commit granularity'.")
    object_type: str = Field(default="", description="Its kind, e.g. 'Preference'.")
    fact: str = Field(description="The full sentence, as you would write it.")


@mcp.tool
async def save_facts(facts: list[Fact], story: str = "", episode_id: str = "",
                     valid_at: datetime | None = None,
                     source: str = "") -> dict:
    """Record one or more facts, optionally with the story behind them.

    Args:
        facts: The relationships to record. Save everything that came out of
            one conversation in a single call so they share a story.
        story: Markdown describing how you came to know this. Optional, but
            it is what you will want when a bare sentence is not enough later.
        episode_id: Attach to an existing story instead of writing a new one —
            use the `episode_id` from a search result when adding to something
            you already recorded.
        valid_at: When these facts became true. Defaults to now; set it for
            something you are recording after the fact.
        source: Where this came from, e.g. 'matrix conversation'.

    Returns:
        The stored `facts` (with their ids) and the `episode_id` they share.
    """
    if not facts:
        return {"facts": [], "episode_id": episode_id or None}
    await store.ensure_indices()

    ep: EpisodicNode | None = None
    if episode_id:
        ep = await EpisodicNode.get_by_uuid(store.driver(), episode_id)
    elif story:
        ep = await store.save_episode(story, source, valid_at)

    saved, uuids = [], []
    for f in facts:
        subj = await store.upsert_entity(
            f.subject, f.subject_type, embedding=await embed.embed(f.subject))
        obj = await store.upsert_entity(
            f.object, f.object_type, embedding=await embed.embed(f.object))
        edge = await store.save_edge(
            subj, f.relation, obj, f.fact, valid_at=valid_at,
            episode_uuid=ep.uuid if ep else None,
            embedding=await embed.embed(f.fact))
        uuids.append(edge.uuid)
        saved.append({"fact_id": edge.uuid, "fact": edge.fact,
                      "subject": subj.name, "relation": edge.name,
                      "object": obj.name,
                      "episode_id": ep.uuid if ep else None,
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
        limit: How many to return.
        include_superseded: Also return facts that have stopped being true —
            use this when you want history rather than the current picture.

    Returns:
        Matching facts, each with its `fact_id`, validity dates, and the
        `episode_id` of the story behind it (pass that to `get_story`).
    """
    await store.ensure_indices()

    # Two ways of being relevant, and they fail differently: vectors miss an
    # exact name they never saw, keywords miss a paraphrase. Run both and
    # merge — a name lookup should not depend on the embedder being reachable.
    found: dict[str, EntityEdge] = {}
    vector = await embed.embed(query)
    if vector is not None:
        for e in await edge_similarity_search(
                store.driver(), vector, None, None, SearchFilters(),
                [store.GROUP_ID], limit):
            found[e.uuid] = e
    try:
        for e in await edge_fulltext_search(
                store.driver(), query, SearchFilters(), [store.GROUP_ID], limit):
            found.setdefault(e.uuid, e)
    except Exception as e:
        logger.warning("Keyword search unavailable: %r", e)

    out = []
    for edge in found.values():
        if edge.invalid_at is not None and not include_superseded:
            continue
        out.append(await _render(edge))
    return out[:limit]


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
async def get_story(episode_id: str) -> dict:
    """Read the story behind a fact — how you came to record it.

    Args:
        episode_id: From a search result or a `save_facts` response.
    """
    ep = await EpisodicNode.get_by_uuid(store.driver(), episode_id)
    return {"episode_id": ep.uuid, "story": ep.content,
            "source": ep.source_description,
            "recorded_at": ep.valid_at.isoformat() if ep.valid_at else None,
            "fact_count": len(ep.entity_edges or [])}


@mcp.tool
async def get_entity(name: str) -> dict:
    """Everything you know about one thing, and how it connects.

    Args:
        name: Exact name, as recorded.

    Returns:
        The entity and every fact it takes part in, in either direction.
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
    """
    return await store.vocabulary()


async def _render(edge: EntityEdge) -> dict:
    src = await EntityNode.get_by_uuid(store.driver(), edge.source_node_uuid)
    tgt = await EntityNode.get_by_uuid(store.driver(), edge.target_node_uuid)
    return {
        "fact_id": edge.uuid,
        "fact": edge.fact,
        "subject": src.name,
        "relation": edge.name,
        "object": tgt.name,
        "valid_at": edge.valid_at.isoformat() if edge.valid_at else None,
        "invalid_at": edge.invalid_at.isoformat() if edge.invalid_at else None,
        "superseded": edge.invalid_at is not None,
        "episode_id": (edge.episodes or [None])[0],
    }


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0",
            port=int(os.environ.get("GRAPH_MCP_PORT", "8005")))
