# mcp_server.py
"""
MCP (Model Context Protocol) server that exposes Repo-Insight's graph tools
as MCP tools. This allows AI coding assistants (Claude Code, Copilot, Cursor)
to natively call graph queries like get_blast_radius before every edit.

Usage:
    python mcp_server.py                    # stdio transport (default)
    python mcp_server.py --transport sse    # SSE transport for web clients

Requires:  pip install mcp
"""

import argparse
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning(
        "MCP SDK not installed. Run: pip install mcp  "
        "The server will not start without it."
    )

from ingest import get_connection
from tools import (
    get_function_context, get_callers, get_callees,
    get_impact_radius, get_blast_radius, get_source_code,
    semantic_search, get_macro_architecture, get_class_architecture,
)


# ---------------------------------------------------------------------------
# Tool definitions (mirroring agent.py TOOL_SCHEMAS)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "get_function_context",
        "description": "Get the definition location, summary, and module for a function by its fully qualified name (FQN).",
        "input_schema": {
            "type": "object",
            "properties": {
                "fqn": {"type": "string", "description": "Fully Qualified Name (e.g., auth.User.login)."}
            },
            "required": ["fqn"],
        },
    },
    {
        "name": "get_callers",
        "description": "Find all functions that directly call the specified function.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fqn": {"type": "string", "description": "FQN of the function to find callers of."}
            },
            "required": ["fqn"],
        },
    },
    {
        "name": "get_callees",
        "description": "Find all functions directly called by the specified function.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fqn": {"type": "string", "description": "FQN of the function to find callees of."}
            },
            "required": ["fqn"],
        },
    },
    {
        "name": "get_impact_radius",
        "description": "Find all functions transitively called BY the function (downstream). Shows what this function touches.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fqn": {"type": "string", "description": "FQN of the function to trace downstream from."}
            },
            "required": ["fqn"],
        },
    },
    {
        "name": "get_blast_radius",
        "description": "Find all functions that transitively CALL the function (upstream). Shows what breaks if this function changes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fqn": {"type": "string", "description": "FQN of the function to assess breakage risk for."}
            },
            "required": ["fqn"],
        },
    },
    {
        "name": "get_source_code",
        "description": "Retrieve the actual source code of a function or class by FQN.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fqn": {"type": "string", "description": "FQN of the function/class to retrieve source code for."}
            },
            "required": ["fqn"],
        },
    },
    {
        "name": "semantic_search",
        "description": "Find functions or classes semantically similar to a natural language query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language description of what to find."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_macro_architecture",
        "description": "Get the high-level macro architecture (Thick Edges) between modules based on calls, imports, and inheritance.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_class_architecture",
        "description": "Get the medium-level architecture within a specific module based on calls and inheritance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "module_name": {"type": "string", "description": "Name of the module (e.g., auth.models)."}
            },
            "required": ["module_name"],
        },
    },
]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_TOOL_MAP = {
    "get_function_context": lambda args, g: get_function_context(fqn=args["fqn"], graph=g),
    "get_callers": lambda args, g: get_callers(fqn=args["fqn"], graph=g),
    "get_callees": lambda args, g: get_callees(fqn=args["fqn"], graph=g),
    "get_impact_radius": lambda args, g: get_impact_radius(fqn=args["fqn"], graph=g),
    "get_blast_radius": lambda args, g: get_blast_radius(fqn=args["fqn"], graph=g),
    "get_source_code": lambda args, g: get_source_code(fqn=args["fqn"], graph=g),
    "semantic_search": lambda args, g: semantic_search(query=args["query"], graph=g),
    "get_macro_architecture": lambda args, g: get_macro_architecture(graph=g),
    "get_class_architecture": lambda args, g: get_class_architecture(module_name=args["module_name"], graph=g),
}


def _dispatch(tool_name: str, arguments: dict[str, Any], graph) -> str:
    """Route an MCP tool call to the corresponding tools.py function."""
    handler = _TOOL_MAP.get(tool_name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        result = handler(arguments, graph)
        return json.dumps(result)
    except Exception as e:
        logger.error("MCP tool '%s' failed: %s", tool_name, e)
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# MCP server setup
# ---------------------------------------------------------------------------

def create_mcp_server() -> "Server":
    """Build and configure the MCP server with all Repo-Insight tools."""
    if not MCP_AVAILABLE:
        raise ImportError(
            "The 'mcp' package is required. Install it with: pip install mcp"
        )

    app = Server("repo-insight")
    graph = get_connection()

    @app.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=td["name"],
                description=td["description"],
                inputSchema=td["input_schema"],
            )
            for td in TOOL_DEFINITIONS
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        result_str = _dispatch(name, arguments, graph)
        return [TextContent(type="text", text=result_str)]

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _run_stdio(app: "Server") -> None:
    """Run the MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repo-Insight MCP Server — expose graph tools to AI coding assistants",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio"],
        default="stdio",
        help="Transport to use (default: stdio)",
    )
    args = parser.parse_args()

    if not MCP_AVAILABLE:
        print("ERROR: MCP SDK not installed. Run: pip install mcp")
        return

    import asyncio
    app = create_mcp_server()
    print("Repo-Insight MCP server starting (stdio transport)...")
    asyncio.run(_run_stdio(app))


if __name__ == "__main__":
    main()
