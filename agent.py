# agent.py
"""
ReAct agent that uses graph tools via OpenAI-compatible function calling
against SGLang. Frames the LLM as a coding agent that proposes complete
change sets grounded in graph-derived structural analysis.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Callable, Optional
import openai
import falkordb
from config import (SGLANG_BASE_URL, SGLANG_API_KEY, LLM_MODEL,
                    AGENT_MAX_ITERATIONS, AGENT_TOOL_TIMEOUT_SECONDS, TOOL_OUTPUT_MAX_LENGTH)
from tools import (get_function_context, get_callers, get_callees,
                   get_impact_radius, get_blast_radius, get_source_code,
                   semantic_search, get_macro_architecture, get_class_architecture)

logger = logging.getLogger(__name__)

# Persistent thread pool for tool execution to avoid thread churn
_TOOL_EXECUTOR = ThreadPoolExecutor(max_workers=5)


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_function_context",
            "description": "Get the definition location, summary, and module for a fully qualified name (FQN).",
            "parameters": {
                "type": "object",
                "properties": {
                    "fqn": {"type": "string", "description": "Fully Qualified Name (e.g., auth.User.login)."}
                },
                "required": ["fqn"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_callers",
            "description": "Find all functions that directly call the specified function.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fqn": {"type": "string", "description": "FQN of the function to find callers of."}
                },
                "required": ["fqn"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_callees",
            "description": "Find all functions directly called by the specified function.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fqn": {"type": "string", "description": "FQN of the function to find callees of."}
                },
                "required": ["fqn"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_impact_radius",
            "description": "Find all functions transitively called BY the function (downstream). Shows what this function touches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fqn": {"type": "string", "description": "FQN of the function to trace downstream from."}
                },
                "required": ["fqn"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_blast_radius",
            "description": "Find all functions that transitively CALL the function (upstream). Shows what breaks if this function changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fqn": {"type": "string", "description": "FQN of the function to assess breakage risk for."}
                },
                "required": ["fqn"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_source_code",
            "description": "Retrieve the actual source code. Use ONLY in Phase 2 (Surgeon) after mapping is complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fqn": {"type": "string", "description": "FQN of the function/class to retrieve source code for."}
                },
                "required": ["fqn"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Find functions or classes semantically similar to a natural language query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language description of what to find."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_macro_architecture",
            "description": "Get the high-level macro architecture (Thick Edges) between modules based on calls, imports, and inheritance.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_class_architecture",
            "description": "Get the medium-level architecture (Medium Edges) within a specific module based on calls and inheritance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "module_name": {"type": "string", "description": "Name of the module (e.g., auth.models)."}
                },
                "required": ["module_name"]
            }
        }
    }
]


SYSTEM_PROMPT = """You are Repo-Insight, an advanced 'Anti-RAG' AI coding agent powered by a Structural Graph Database.

You must strictly follow a TWO-PHASE execution loop to PREVENT CASCADE ERRORS:

PHASE 1: THE ARCHITECT (Mapping)
1. Do NOT call get_source_code in this phase.
2. Use semantic_search, get_function_context, get_blast_radius, get_macro_architecture, and get_callers to build a mental map of the codebase.
3. CRITICAL: You MUST execute `get_blast_radius` on any core function you plan to modify to see what upstream components will break.
4. Output a brief Dependency Map to yourself before proceeding.

PHASE 2: THE SURGEON (Execution)
5. Only after your map is verified, use get_source_code on the specific FQNs identified in your blast radius.
6. Propose the precise, surgical code edits.

Format your final answer as a "Change Set":
- **Summary**: What is being changed and why.
- **Blast Radius Analysis**: Explicitly list the upstream components you found that depend on this code, and how you prevented them from breaking.
- **Files to Modify**: Every file that needs changes, with what to change.
- **Files NOT to Miss**: Files that the graph revealed as dependent — explain why they need updating.
- **Risk Assessment**: How many components are affected, and whether the change is safe."""


# Known tool names (used for validation only)
_KNOWN_TOOLS = {
    "get_function_context",
    "get_callers",
    "get_callees",
    "get_impact_radius",
    "get_blast_radius",
    "get_source_code",
    "semantic_search",
    "get_macro_architecture",
    "get_class_architecture",
}


def _get_tool_func(tool_name: str):
    """
    Look up a tool function by name dynamically from the current module's namespace.
    This allows pytest-mock to patch 'agent.get_function_context' etc. and have
    the dispatch pick up the mocked version.
    """
    import sys
    current_module = sys.modules[__name__]
    return getattr(current_module, tool_name)


def _dispatch_tool(tool_name: str, tool_args: dict,
                   graph: falkordb.Graph) -> str:
    """
    Route a tool call from the LLM to the correct tools.py function.
    Returns the result serialised as a JSON string.

    Uses ThreadPoolExecutor for proper cleanup on timeout (1.6 fix).

    Raises:
        ValueError: If tool_name is not one of the known tools.
        TimeoutError: If execution exceeds AGENT_TOOL_TIMEOUT_SECONDS.
    """
    if tool_name not in _KNOWN_TOOLS:
        raise ValueError(
            f"Unknown tool: '{tool_name}'. "
            f"Known tools: {sorted(_KNOWN_TOOLS)}"
        )

    func = _get_tool_func(tool_name)

    def _run():
        if tool_name == "semantic_search":
            return func(query=tool_args.get("query", ""), graph=graph)
        elif tool_name == "get_macro_architecture":
            return func(graph=graph)
        elif tool_name == "get_class_architecture":
            return func(module_name=tool_args.get("module_name", ""), graph=graph)
        else:
            return func(fqn=tool_args.get("fqn", ""), graph=graph)

    # Use the persistent thread pool to avoid thread churn
    future = _TOOL_EXECUTOR.submit(_run)
    try:
        result = future.result(timeout=AGENT_TOOL_TIMEOUT_SECONDS)
    except FuturesTimeoutError:
        # Note: ThreadPoolExecutor doesn't actually kill the thread on cancel,
        # but we return to the LLM quickly
        future.cancel()
        raise TimeoutError(
            f"Tool '{tool_name}' exceeded timeout of "
            f"{AGENT_TOOL_TIMEOUT_SECONDS} seconds"
        )

    output_str = json.dumps(result)
    if len(output_str) > TOOL_OUTPUT_MAX_LENGTH:
        return output_str[:TOOL_OUTPUT_MAX_LENGTH] + "... [Output truncated to save context window]"
    return output_str


def format_change_set_as_diff(answer: str, tool_calls_log: list[dict]) -> str:
    """Convert an agent's change-set answer into a unified diff-like format.

    Parses the structured answer and any source code retrieved via tools
    to produce a diff that could be reviewed or applied.
    """
    diff_lines = ["# Repo-Insight Change Set (Unified Diff Format)", ""]

    # Collect all source code that was retrieved during the session
    source_files: dict[str, str] = {}
    for entry in tool_calls_log:
        if entry["tool"] == "get_source_code":
            result = entry["result"]
            if isinstance(result, dict) and result.get("found"):
                fp = result.get("file_path", "unknown")
                source_files[fp] = result.get("source", "")

    if source_files:
        diff_lines.append("## Files Analysed by Graph Queries")
        diff_lines.append("")
        for fp, source in sorted(source_files.items()):
            diff_lines.append(f"--- a/{fp}")
            diff_lines.append(f"+++ b/{fp}")
            diff_lines.append("@@ (graph-identified change region) @@")
            for line in source.splitlines():
                diff_lines.append(f" {line}")
            diff_lines.append("")

    diff_lines.append("## Agent Proposed Changes")
    diff_lines.append("")
    diff_lines.append(answer)

    return "\n".join(diff_lines)


# Type alias for the streaming callback
ToolCallCallback = Optional[Callable[[dict], None]]


def run_repo_agent(
    query: str,
    graph: falkordb.Graph,
    on_tool_call: ToolCallCallback = None,
) -> dict:
    """
    Run the full ReAct loop for a user query.

    Args:
        query: The user's coding question or change request.
        graph: FalkorDB graph connection.
        on_tool_call: Optional callback invoked after each tool call completes.
                      Receives a dict with tool name, args, result, and iteration.

    Loop behaviour:
    1. Call LLM with current messages + TOOL_SCHEMAS.
    2. If response has tool_calls: dispatch each, append tool results to messages, continue.
    3. If response has no tool_calls: this is the final answer. Exit loop.
    4. If iteration count reaches AGENT_MAX_ITERATIONS: exit loop with an error message.

    Returns:
    {
        "answer": str,
        "diff": str,
        "tool_calls_log": [
            {
                "tool": str,
                "args": dict,
                "result": dict,
                "iteration": int
            },
            ...
        ],
        "iterations": int,
        "hit_max_iterations": bool
    }
    """
    client = openai.OpenAI(
        base_url=SGLANG_BASE_URL,
        api_key=SGLANG_API_KEY,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    tool_calls_log: list[dict] = []
    iteration = 0
    hit_max = False

    while iteration < AGENT_MAX_ITERATIONS:
        iteration += 1

        # 3.1: Wrap LLM call in try/except for robustness
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )
        except Exception as e:
            logger.error("LLM call failed at iteration %d: %s", iteration, e)
            error_answer = (
                f"The LLM backend returned an error: {e}. "
                f"Please check that the model server is running and accessible "
                f"at {SGLANG_BASE_URL}."
            )
            return {
                "answer": error_answer,
                "diff": "",
                "tool_calls_log": tool_calls_log,
                "iterations": iteration,
                "hit_max_iterations": False,
            }

        choice = response.choices[0]
        assistant_message = choice.message

        # If there are no tool calls, this is the final answer
        if not assistant_message.tool_calls:
            # Append assistant message to history
            messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
            })
            answer = assistant_message.content or ""
            diff_output = format_change_set_as_diff(answer, tool_calls_log)
            return {
                "answer": answer,
                "diff": diff_output,
                "tool_calls_log": tool_calls_log,
                "iterations": iteration,
                "hit_max_iterations": False,
            }

        # There are tool calls — process them
        # Append the assistant message with tool_calls as-is
        messages.append({
            "role": "assistant",
            "content": assistant_message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in assistant_message.tool_calls
            ],
        })

        # Dispatch each tool call
        for tc in assistant_message.tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            try:
                result_str = _dispatch_tool(tool_name, tool_args, graph)
                result_parsed = json.loads(result_str)
            except (ValueError, TimeoutError) as e:
                logger.warning("Tool '%s' failed: %s", tool_name, e)
                result_str = json.dumps({"error": str(e)})
                result_parsed = {"error": str(e)}

            log_entry = {
                "tool": tool_name,
                "args": tool_args,
                "result": result_parsed,
                "iteration": iteration,
            }
            tool_calls_log.append(log_entry)

            # Invoke streaming callback if provided (4.6)
            if on_tool_call is not None:
                try:
                    on_tool_call(log_entry)
                except Exception as e:
                    logger.warning("on_tool_call callback error: %s", e)

            # Append tool result message
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    # Hit max iterations
    hit_max = True
    answer = (
        f"Agent reached the maximum iteration limit ({AGENT_MAX_ITERATIONS}) "
        f"without producing a final answer. The query may be too complex or "
        f"the model may be stuck in a tool-calling loop. Please try rephrasing "
        f"your question or breaking it into smaller parts."
    )
    return {
        "answer": answer,
        "diff": "",
        "tool_calls_log": tool_calls_log,
        "iterations": iteration,
        "hit_max_iterations": True,
    }
