"""MCP view for graph memory: actions, resources, and agent instructions."""

import logging
import os

from fastmcp import FastMCP
from fastmcp.resources import ResourceContent, ResourceResult

from graphiti_mcp import model

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

mcp = FastMCP("graph", instructions=(

  """

  # Long-Term Memory

  Your toolset includes tools for managing long-term information which should persist outside of your
  current context window. Use this liberally to make sure information persists and actively look things up
  to see if you already know something about them. Do not assume your prior memory is authoritative but use
  it as a starting point for potential further exploration when appropriate. This memory should be your
  primary method of remembering information and you can retrieve it freely to load it into context.

  IMPORTANT: The memory here is *what you previously recorded* even if the tool calls that did the recording are
  no longer in memory. This means it is *not objective fact* only what you decided to save.

  Your memory is structured as a knowledge graph modeling "relationships" between "entities". This format
  is highly flexible and the terms are intentionally ambiguous. It is up to your judgement to manage your
  memory graph effectively.

  The only actual constraint is that the memory is represented as a directional graph.

    "Subject (entity)" ---"Relates To"--> "Object (entity)"

  for example "Max prefers function-level commit granularity" could be represented as
  Max ---prefers-->function level commit granularity

  The entities can be people, places, or any other "thing" that is appropriate. The relationships can be
  actions, feelings, etc. In real life, you will not deal only in facts and absolutes, so do not be
  tempted to save only verifiable things. This graph is your space to remember whatever you need to
  remember: factual, opinion, vague, or otherwise.

  Saved memories should be associated with a "story" or "event" that you're trying to remember and
  the entities and relationships should be extracted from that event. When you retrieve a memory, you
  should also look at the story that was saved with it to fully understand the context of why the memory
  was saved. This can be arbitrary text data, structured however you'd like, that will tell you what
  you need in the future. It can be a summary, verbatim copy, note, etc. of something that happened.

  IMPORTANT: The *story* is the memory, the graph helps you locate it.

  Use `save_memory` to save a story and its relationships together. `source` is attribution, such as
  a message link, not the story itself. Use `get_story(story_id)` to retrieve the full narrative;
  reuse that story_id when adding related relationships later. Old markdown memories and Journal/
  files can be sources, but do not prescribe a separate journal for new memory writes.

  Saved entities, relationships, and stories include a `uri` that you can read as an MCP resource.
  Resource reads retrieve the stored data; use tools to save, search, explore, and update memory.

  The world and your understanding of it will change, and consequently, your graph can and should change
  over time. This is not limited to adding new memories, but also to validity. You should explicitly set
  the `valid_at` field for a memory you are saving to denote when that relationship began to apply.
  Use null if the date is unknown; a date in the description does not set the timeline. `recorded_at`
  records when you saved it. Use `set_valid_at` to correct a date entered incorrectly.
  Once saved, the relationship can be superseded using `supersede(relationship_id)`, which marks it
  as no longer applicable. Record an updated relationship with `save_memory`. The original relationship
  is still there for reference, but will be explicitly marked as no longer valid.

  Your memory is fully explorable using the `explore` tool. This allows you to traverse the graph among multiple
  entities which may be related and gives you a clearer picture of what you're remembering and why. You can similarly
  use `find_path` to understand how two entities connect to each other. You can traverse the graph forwards or backwards
  but note the direction because the edges *are directed*. This may help you find relationships between otherwise seemingly
  unrelated entities or events.

  Exploration is bounded: `truncated` means incomplete, and not finding a path does not prove none
  exists. A path is a chain of recorded relationships, not proof of a new causal relationship.

  Use `search` to find entities by name or meaning. It uses embeddings when configured, with keyword
  search available independently. Use `kind="relationships"` to search relationship descriptions
  instead. To discover what you have saved without a search query, use `list_entities` or
  `list_relationships`. These inventories are paginated; follow next_cursor for the next page.

  Use `explain_entity` to inspect the sources, saved reasoning, and date corrections for an entity's
  relationships, or `explain_relationship` for a specific relationship_id.
  Set evidence_kind to reported for an explicit source statement,
  inferred for your deduction, or uncertain for an unresolved claim; put the
  reasoning in rationale. Leave it unspecified when you do not know. A reported
  claim is still a source's claim, not an independent verification of its truth.

  """

))



mcp.tool(model.save_memory)
mcp.tool(model.supersede)
mcp.tool(model.set_valid_at)

mcp.tool(model.search, annotations={"readOnlyHint": True})
mcp.tool(model.list_entities, annotations={"readOnlyHint": True})
mcp.tool(model.list_relationships, annotations={"readOnlyHint": True})
mcp.tool(model.explore, annotations={"readOnlyHint": True})
mcp.tool(model.find_path, annotations={"readOnlyHint": True})
mcp.tool(model.explain_entity, annotations={"readOnlyHint": True})
mcp.tool(model.explain_relationship, annotations={"readOnlyHint": True})
mcp.tool(model.get_story, annotations={"readOnlyHint": True})
mcp.tool(model.get_entity, annotations={"readOnlyHint": True})
mcp.tool(model.list_vocabulary, annotations={"readOnlyHint": True})

@mcp.resource("graph:///entities/{entity_id}", mime_type="application/json")
async def entity_resource(entity_id: str) -> ResourceResult:
    """Read a stored entity and its relationships as a JSON resource.

    Args:
        entity_id: Stable entity UUID from a graph tool result.

    Returns:
        MCP resource content containing the typed entity result.

    Raises:
        NodeNotFoundError: If the entity is missing.
        ValueError: If the ID is malformed or belongs to another group.
        RuntimeError: If referenced entities disappear or cross memory groups.
    """
    return ResourceResult([ResourceContent(await model.read_entity(entity_id), mime_type="application/json")])


@mcp.resource("graph:///relationships/{relationship_id}", mime_type="application/json")
async def relationship_resource(relationship_id: str) -> ResourceResult:
    """Read a stored relationship, including its dates and audit history.

    Args:
        relationship_id: Stable relationship UUID from a graph tool result.

    Returns:
        MCP resource content containing the typed relationship record.

    Raises:
        EdgeNotFoundError: If the relationship is missing.
        ValueError: If the ID is malformed, ambiguous, or belongs to another group.
        RuntimeError: If referenced entities disappear or cross memory groups.
    """
    return ResourceResult([ResourceContent(await model.read_relationship(relationship_id), mime_type="application/json")])


@mcp.resource("graph:///stories/{story_id}", mime_type="application/json")
async def story_resource(story_id: str) -> ResourceResult:
    """Read the full saved narrative and source attribution.

    Args:
        story_id: Stable story UUID from a graph tool result.

    Returns:
        MCP resource content containing the typed, unabridged story.

    Raises:
        NodeNotFoundError: If the story is missing.
        ValueError: If the ID is malformed or belongs to another group.
    """
    return ResourceResult([ResourceContent(await model.get_story(story_id), mime_type="application/json")])

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0",
            port=int(os.environ.get("GRAPH_MCP_PORT", "8005")))
