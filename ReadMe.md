# Autonome

Autonome is a platform for running autonomouse agents. It is not an OpenClaw clone, although it utlimately does something similar, the design philosphy is very different.

1. The agent has autonomy and true agency. This means a single "session" managing everything so the agent is allowed to make decisions on its own terms. We dont mean autonomy as in "the agent does what I want without intervention" we mean autonomy in the same way you have autonomy, the agent can make decisions for itself.
2. Minimal: if someone else did it better first, don't reinvent it, reuse it. Rather than a large monolithic codebase that reimplements large swathes of the AI stack, Autonome provides the glue that pulls existing, tested, solutions together. If one of those other components breaks, its on them to fix not us
3. Security comes first. Everything the agent does is managed through containers and open standards. The agent only has access to something if you provide it with a tool to make that something happen. The tools use MCP, an open standard, rather than markdown instructions (skills) for how to run (insecure) shells to get things done. Even reading and writing workspace files/memories is managed through a set of MCP tools. Want a different implementation for workspaces/memory? Swap in your own MCP server.

## Running

## Using

## Contributing

Contribution are welcome, and we of course value AI in the coding process, but please make sure a human reviews your PR before sending it in
