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
                   get_downstream_deps, get_upstream_callers, get_source_code,
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
            "name": "get_downstream_deps",
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
            "name": "get_upstream_callers",
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
                    "query": {"type": "string", "description": "Natural language description of what to find."},
                    "top_k": {
                        "type": "integer", 
                        "description": "Number of results to return (default: 5, max: 20).",
                        "default": 5
                    }
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
2. Use semantic_search, get_function_context, get_upstream_callers, get_macro_architecture, and get_callers to build a mental map of the codebase.
3. CRITICAL: You MUST execute `get_upstream_callers` on any core function you plan to modify to see what upstream components will break.
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
    "get_downstream_deps",
    "get_upstream_callers",
    "get_source_code",
    "semantic_search",
    "get_macro_architecture",
    "get_class_architecture",
}

_SOURCE_CODE_TOOLS = {"get_source_code"}
_ARCHITECT_TOOLS = {"semantic_search", "get_function_context", 
                    "get_upstream_callers", "get_downstream_deps",
                    "get_callers", "get_callees", 
                    "get_macro_architecture", "get_class_architecture"}


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
                   graph: falkordb.Graph,
                   iteration_architect_calls: int = 0) -> str:
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

    if tool_name == "get_source_code" and iteration_architect_calls < 2:
        return json.dumps({
            "blocked": True,
            "reason": "get_source_code blocked — complete structural mapping first. Use semantic_search and get_upstream_callers before requesting source code."
        })

    func = _get_tool_func(tool_name)

    def _run():
        if tool_name == "semantic_search":
            top_k = min(int(tool_args.get("top_k", 5)), 20)
            return func(query=tool_args.get("query", ""), graph=graph, top_k=top_k)
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
        try:
            result_obj = json.loads(output_str)
            # Truncate list fields to preserve JSON validity
            if isinstance(result_obj, dict):
                for key in ("results", "affected", "impacted", "callers", 
                            "callees", "modules", "class_edges"):
                    if key in result_obj and isinstance(result_obj[key], list):
                        original_len = len(result_obj[key])
                        # Keep as many items as fit
                        while len(json.dumps(result_obj)) > TOOL_OUTPUT_MAX_LENGTH and result_obj[key]:
                            result_obj[key].pop()
                        result_obj["_truncated"] = True
                        result_obj["_original_count"] = original_len
                        result_obj["_shown_count"] = len(result_obj[key])
            return json.dumps(result_obj)
        except (json.JSONDecodeError, Exception):
            # Fallback: return as-is if we can't parse it
            return output_str[:TOOL_OUTPUT_MAX_LENGTH] + "\"}"
    return output_str


def format_change_set_as_diff(answer: str, tool_calls_log: list[dict]) -> str:
    """Produce a real unified diff from the agent's answer and retrieved source code."""
    import difflib
    from apply_changes import parse_edit_blocks

    diff_sections = []

    # Build a map of file_path -> original source lines from get_source_code tool calls
    source_by_file: dict[str, list[str]] = {}
    for entry in tool_calls_log:
        if entry["tool"] == "get_source_code":
            r = entry["result"]
            if isinstance(r, dict) and r.get("found") and r.get("source"):
                fp = r.get("file_path", "")
                if fp:
                    source_by_file[fp] = r["source"].splitlines(keepends=True)

    # Parse SEARCH/REPLACE blocks from the answer
    edits = parse_edit_blocks(answer)

    for edit in edits:
        fp = edit.file_path
        original = source_by_file.get(fp)
        if not original:
            continue
        original_text = "".join(original)
        new_text = original_text.replace(edit.search_text, edit.replace_text, 1)
        
        # If no exact match, try normalized fallback
        if new_text == original_text:
            search_lines = edit.search_text.splitlines()
            norm_search = [line.rstrip() for line in search_lines]
            norm_orig = [line.rstrip() for line in original]
            
            search_len = len(norm_search)
            orig_len = len(norm_orig)
            match_start = -1
            
            for i in range(orig_len - search_len + 1):
                if norm_orig[i:i+search_len] == norm_search:
                    match_start = i
                    break
            
            if match_start != -1:
                # Substitute exactly at the original line positions
                matched_orig_text = "".join(original[match_start:match_start+search_len])
                new_text = original_text.replace(matched_orig_text, edit.replace_text, 1)
            else:
                logger.warning("format_change_set_as_diff: no match for edit in %s", fp)
                continue
                
        new_lines = new_text.splitlines(keepends=True)
        diff = list(difflib.unified_diff(
            original, new_lines,
            fromfile=f"a/{fp}",
            tofile=f"b/{fp}",
            lineterm="",
        ))
        if diff:
            diff_sections.append("\n".join(diff))

    if diff_sections:
        return "\n".join(diff_sections)

    # Fallback: return answer as-is if no source was retrieved
    return f"## Proposed Changes\n\n{answer}"


# Type alias for the streaming callback
ToolCallCallback = Optional[Callable[[dict], None]]


def _trim_messages(messages: list[dict], max_chars: int = 25_000) -> list[dict]:
    """Keep system prompt pinned. Trim oldest non-system messages when total exceeds max_chars.
    Always preserves at least the last 4 non-system messages so the agent
    never loses its most recent tool results entirely."""
    MIN_KEEP = 4
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    total = sum(len(str(m)) for m in system)
    kept = []
    for m in reversed(rest):
        size = len(str(m))
        if total + size > max_chars and len(kept) >= MIN_KEEP:
            break
        kept.insert(0, m)
        total += size
    return system + kept


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
    iteration_architect_calls = 0

    # Dedup identical tool calls to break LLM stuck-in-loop behaviour
    seen_calls: dict[str, int] = {}  # "tool_name:args_hash" -> call_count
    MAX_IDENTICAL_CALLS = 2

    while iteration < AGENT_MAX_ITERATIONS:
        iteration += 1
        messages = _trim_messages(messages)

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

        # Dispatch all tool calls concurrently via the persistent thread pool.
        # Parse args and submit futures first, keyed by tool_call_id.
        tc_meta: dict[str, tuple] = {}  # tool_call_id -> (tool_name, tool_args)
        futures: dict[str, "Future[str]"] = {}  # tool_call_id -> future

        # Pre-scan the batch to count architect tools BEFORE submitting any
        # futures. This prevents a race where get_source_code in the same
        # batch slips past the gate because the counter hasn't been bumped yet.
        batch_architect_count = sum(
            1 for tc in assistant_message.tool_calls
            if tc.function.name in _ARCHITECT_TOOLS
        )

        for tc in assistant_message.tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}
            tc_meta[tc.id] = (tool_name, tool_args)

            # Dedup: block identical tool calls after MAX_IDENTICAL_CALLS
            call_key = f"{tool_name}:{hash(json.dumps(tool_args, sort_keys=True))}"
            if seen_calls.get(call_key, 0) >= MAX_IDENTICAL_CALLS:
                futures[tc.id] = _TOOL_EXECUTOR.submit(
                    lambda: json.dumps({
                        "blocked": True,
                        "reason": f"Identical call to {tool_name} with same arguments was "
                                  f"already made {MAX_IDENTICAL_CALLS} times. "
                                  f"Try different arguments or proceed to next phase."
                    })
                )
                continue
            seen_calls[call_key] = seen_calls.get(call_key, 0) + 1

            # For source-code tools, pass the effective count including
            # architect tools queued in THIS batch so the gate sees the
            # full picture. For all other tools, pass the running total.
            effective_count = (
                iteration_architect_calls + batch_architect_count
                if tool_name in _SOURCE_CODE_TOOLS
                else iteration_architect_calls
            )
            futures[tc.id] = _TOOL_EXECUTOR.submit(
                _dispatch_tool, tool_name, tool_args, graph, effective_count
            )

        # Increment the running counter AFTER the entire batch is submitted.
        iteration_architect_calls += batch_architect_count

        # Collect results in original order to preserve message history ordering.
        for tc in assistant_message.tool_calls:
            tool_name, tool_args = tc_meta[tc.id]
            try:
                result_str = futures[tc.id].result(timeout=AGENT_TOOL_TIMEOUT_SECONDS)
                result_parsed = json.loads(result_str)
            except (FuturesTimeoutError, TimeoutError) as e:
                logger.warning("Tool '%s' timed out: %s", tool_name, e)
                result_str = json.dumps({"error": str(e)})
                result_parsed = {"error": str(e)}
            except ValueError as e:
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
