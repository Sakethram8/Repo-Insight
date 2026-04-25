# agent.py
"""
ReAct agent that uses graph tools via OpenAI-compatible function calling
against SGLang. Frames the LLM as a coding agent that proposes complete
change sets grounded in graph-derived structural analysis.
"""

import json
import time
import threading
import openai
import falkordb
from config import (SGLANG_BASE_URL, SGLANG_API_KEY, LLM_MODEL,
                    AGENT_MAX_ITERATIONS, AGENT_TOOL_TIMEOUT_SECONDS)
from tools import (get_function_context, get_callers, get_callees,
                   get_impact_radius, get_blast_radius, get_source_code,
                   semantic_search)


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

You must strictly follow a TWO-PHASE execution loop:

PHASE 1: THE ARCHITECT (Mapping)
1. Do NOT call get_source_code in this phase.
2. Use semantic_search, get_function_context, get_blast_radius, get_macro_architecture, and get_callers to build a mental map of the codebase.
3. Understand the Fully Qualified Names (FQNs) of the nodes involved, and read their AI-generated summaries.
4. Output a brief Dependency Map to yourself before proceeding.

PHASE 2: THE SURGEON (Execution)
5. Only after your map is verified, use get_source_code on the specific FQNs identified in your blast radius.
6. Propose the precise, surgical code edits.

Format your final answer as a "Change Set":
- **Summary**: What is being changed and why.
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

    Raises:
        ValueError: If tool_name is not one of the five known tools.
        TimeoutError: If execution exceeds AGENT_TOOL_TIMEOUT_SECONDS.
    """
    if tool_name not in _KNOWN_TOOLS:
        raise ValueError(
            f"Unknown tool: '{tool_name}'. "
            f"Known tools: {sorted(_KNOWN_TOOLS)}"
        )

    func = _get_tool_func(tool_name)
    result = [None]
    error = [None]

    def _run():
        try:
            if tool_name == "semantic_search":
                result[0] = func(query=tool_args.get("query", ""), graph=graph)
            elif tool_name == "get_macro_architecture":
                result[0] = func(graph=graph)
            elif tool_name == "get_class_architecture":
                result[0] = func(module_name=tool_args.get("module_name", ""), graph=graph)
            else:
                result[0] = func(fqn=tool_args.get("fqn", ""), graph=graph)
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=_run)
    thread.start()
    thread.join(timeout=AGENT_TOOL_TIMEOUT_SECONDS)

    if thread.is_alive():
        raise TimeoutError(
            f"Tool '{tool_name}' exceeded timeout of "
            f"{AGENT_TOOL_TIMEOUT_SECONDS} seconds"
        )

    if error[0] is not None:
        raise error[0]

    return json.dumps(result[0])


def run_repo_agent(query: str, graph: falkordb.Graph) -> dict:
    """
    Run the full ReAct loop for a user query.

    Loop behaviour:
    1. Call LLM with current messages + TOOL_SCHEMAS.
    2. If response has tool_calls: dispatch each, append tool results to messages, continue.
    3. If response has no tool_calls: this is the final answer. Exit loop.
    4. If iteration count reaches AGENT_MAX_ITERATIONS: exit loop with an error message.

    Returns:
    {
        "answer": str,
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

    tool_calls_log = []
    iteration = 0
    hit_max = False

    while iteration < AGENT_MAX_ITERATIONS:
        iteration += 1

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )

        choice = response.choices[0]
        assistant_message = choice.message

        # If there are no tool calls, this is the final answer
        if not assistant_message.tool_calls:
            # Append assistant message to history
            messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
            })
            return {
                "answer": assistant_message.content or "",
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
                result_str = json.dumps({"error": str(e)})
                result_parsed = {"error": str(e)}

            tool_calls_log.append({
                "tool": tool_name,
                "args": tool_args,
                "result": result_parsed,
                "iteration": iteration,
            })

            # Append tool result message
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    # Hit max iterations
    hit_max = True
    return {
        "answer": (
            f"Agent reached the maximum iteration limit ({AGENT_MAX_ITERATIONS}) "
            f"without producing a final answer. The query may be too complex or "
            f"the model may be stuck in a tool-calling loop. Please try rephrasing "
            f"your question or breaking it into smaller parts."
        ),
        "tool_calls_log": tool_calls_log,
        "iterations": iteration,
        "hit_max_iterations": True,
    }
