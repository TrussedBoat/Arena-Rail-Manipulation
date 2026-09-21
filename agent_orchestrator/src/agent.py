import json
from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from config import get_runtime_config
from tools import (
    execute_pick_script,
    execute_place_script,
    move_rail_to_object,
    scan_object,
    general_mapping,
    targeted_search,
    start_joint_controller,
    turn_panda_arm,
    home_panda_arm,
    move_rail_relative,
    verify_settled_camera_frame,
    search_semantic_objects,
)

runtime_config = get_runtime_config()
MODEL_NAME = runtime_config.vlm.model_alias

tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "start_joint_controller",
            "description": (
                "Initialize the robot at its current rail position, set the arm to its "
                "safe startup pose, and open the gripper unless it is already holding "
                "a previously confirmed object. Always call this first."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_semantic_objects",
            "description": "Natural-language semantic-map lookup powered by MobileCLIP text-to-image similarity. Use for visual descriptions or non-canonical wording such as 'purple bowl', 'small red apple', or 'remote control'. Returns at most five confirmed compatible tracks ranked by cosine_similarity; it is map evidence, not visual proof.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Short natural-language description of the desired physical object, not the full task request.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "targeted_search",
            "description": (
                "Find one canonical class and refresh its semantic location with an RRT scan. "
                "Choose it when exactly one missing, stale, candidate, or ambiguous class is "
                "enough to continue the task. Pass a base detector class such as 'apple'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_object": {
                        "type": "string",
                        "description": "Normalized target label, for example 'apple'.",
                    }
                },
                "required": ["target_object"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "general_mapping",
            "description": (
                "Build a full semantic map: moves to the minimum, centre, and maximum rail "
                "stations and uses RRT wrist-camera arcs to scan both desk rows. Semantic "
                "perception records all objects during the complete route. Choose it for a "
                "spatial/relational task whose needed context is missing, or when two or more "
                "relevant classes are absent or not confirmed."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_pick_script",
            "description": "Pick one exact confirmed semantic object after move_rail_to_object(object_id) and camera verification. Uses staged RRT hover, descend, close, and retreat motion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_id": {
                        "type": "string",
                        "description": "Exact object_id returned by semantic map retrieval.",
                    }
                },
                "required": ["object_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_rail_to_object",
            "description": "Rail-align and point at one selected confirmed semantic object ID returned by semantic map retrieval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_id": {
                        "type": "string",
                        "description": "Exact confirmed object_id, or home/start.",
                    }
                },
                "required": ["object_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_object",
            "description": (
                "Perform a close object-centered semantic scan after "
                "move_rail_to_object(object_id). The wrist camera follows a configurable "
                "270-degree RRT arc at fixed standoff while semantic perception continues "
                "processing RGB-D frames and refining the same stored object track."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "object_id": {
                        "type": "string",
                        "description": "Exact confirmed object_id that was passed to move_rail_to_object.",
                    }
                },
                "required": ["object_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_rail_relative",
            "description": "Move the robot's base rail by a specific relative distance in meters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "relative_distance_m": {
                        "type": "number",
                        "description": "The distance in meters to move. Positive moves forward, negative moves backward.",
                    }
                },
                "required": ["relative_distance_m"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_place_script",
            "description": "Place the currently held item above one exact confirmed destination object after move_rail_to_object(destination_object_id) and camera verification. Uses staged RRT hover, descend, open, and retreat motion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination_object_id": {
                        "type": "string",
                        "description": "Exact confirmed destination object_id returned by semantic map retrieval.",
                    }
                },
                "required": ["destination_object_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "turn_panda_arm",
            "description": "Turn the robot's arm to a specific angle in radians (e.g., 3.14 for a 180-degree turn, or 1.57 for 90 degrees).",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_rad": {
                        "type": "number",
                        "description": "The target angle in radians.",
                    }
                },
                "required": ["target_rad"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "home_panda_arm",
            "description": "Return the robot's arm to its default forward-facing 0.0 rad position.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_camera_frame",
            "description": "Wait two seconds for the robot/camera to settle, capture two new ROS wrist-camera frames, then ask an isolated context-free VLM to answer one narrow visual question about the later frame. The planner receives structured evidence only, never the raw image. Call immediately before every execute_pick_script or execute_place_script. Do not manipulate unless target_confirmed and safe_for_action are both true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "minLength": 3,
                        "maxLength": 300,
                        "description": "A short, image-grounded question naming the specific object and required action, e.g. 'Is the red apple clearly visible and safely graspable?'",
                    },
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_task",
            "description": "Complete the workflow once the user's objective is achieved and the robot has safely returned home.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "One short completion summary.",
                    }
                },
                "required": ["summary"],
                "additionalProperties": False,
            },
        },
    }
]

for t in tool_definitions:
    t["function"]["parameters"]["properties"]["thought"] = {
        "type": "string",
        "description": "Mandatory step-by-step logical reasoning and verification before taking this action. Output your thoughts wrapped in <think> tags."
    }
    t["function"]["parameters"]["required"].append("thought")
llm = ChatOpenAI(
    base_url=runtime_config.vlm.api_base_url,
    api_key="not-needed",
    model=MODEL_NAME,
    temperature=0.2,
    timeout=runtime_config.vlm.request_timeout_sec,
    max_tokens=runtime_config.vlm.max_completion_tokens,
).bind_tools(tool_definitions)

class AgentState(TypedDict, total=False):
    messages: list[BaseMessage]
    iterations: int
    terminated: bool
    outcome: str
    failure_reason: str
    tools_called: list[str]
    execution_ledger: list[str]
    semantic_query_cache: dict[str, str]
    last_visual_verification: dict
    last_navigation_object_id: str
BOWL_TARGETS = {"bowl", "purple bowl"}


tools_impl = {
    "start_joint_controller": lambda _: start_joint_controller(),
    "general_mapping": lambda _: general_mapping(),
    "targeted_search": lambda args: targeted_search(
        args.get("target_object")
    ),
    "execute_pick_script": lambda args: execute_pick_script(
        args.get("object_id")
    ),
    "move_rail_to_object": lambda args: move_rail_to_object(
        args.get("object_id")
    ),
    "scan_object": lambda args: scan_object(args.get("object_id")),
    "move_rail_relative": lambda args: move_rail_relative(
        args.get("relative_distance_m")
    ),
    "execute_place_script": lambda args: execute_place_script(
        args.get("destination_object_id")
    ),
    "turn_panda_arm": lambda args: turn_panda_arm(
        args.get("target_rad")
    ),
    "home_panda_arm": lambda _: home_panda_arm(),
    "get_camera_frame": lambda args: verify_settled_camera_frame(
        args.get("question", ""), timeout_sec=10.0, settle_sec=2.0, frame_count=2
    ),
    "search_semantic_objects": lambda args: search_semantic_objects(args.get("query", "")),
    "finish_task": lambda args: {
        "status": "success",
        "success": True,
        "summary": args.get("summary"),
    },
}


def _normalized_argument(arguments: dict, name: str) -> str:
    return str(arguments.get(name) or "").strip().lower()


def _validate_tool_for_stage(
    state: AgentState, tool_name: str, arguments: dict
) -> str | None:
    tools_called = state.get("tools_called", [])
    
    if len(tools_called) == 0 and tool_name != "start_joint_controller":
        return "The very first tool call must be 'start_joint_controller'."
        
    if tool_name == "finish_task":
        if not str(arguments.get("summary") or "").strip():
            return "finish_task requires a non-empty summary."

    target = _normalized_argument(arguments, "target_object")
    if tool_name == "targeted_search" and not target:
        return "Search requires a non-empty canonical pickup target."

    if tool_name == "scan_object":
        object_id = str(arguments.get("object_id") or "").strip()
        if not object_id:
            return "scan_object requires a confirmed object_id."
        if (
            not tools_called
            or tools_called[-1] != "move_rail_to_object"
            or object_id != state.get("last_navigation_object_id", "")
        ):
            return (
                "Call move_rail_to_object with this exact object_id immediately before "
                "starting its object-centered scan."
            )

    if tool_name in ("execute_pick_script", "execute_place_script"):
        if not tools_called or tools_called[-1] != "get_camera_frame":
            return (
                "Immediately before every pick or place hardware execution, call "
                "get_camera_frame with a narrow visual question."
            )
        verification = state.get("last_visual_verification", {})
        if (
            not isinstance(verification, dict)
            or verification.get("target_confirmed") is not True
            or verification.get("safe_for_action") is not True
        ):
            return (
                "The latest isolated visual verification did not clearly confirm a "
                "safe target. Reposition or capture another frame; do not manipulate."
            )

    return None


def _tool_result_succeeded(tool_name: str, result: object) -> bool:
    if isinstance(result, dict):
        if tool_name in ("targeted_search", "general_mapping"):
            return result.get("status") == "success" and result.get("success") is True
        if "success" in result:
            return result.get("success") is True
        return result.get("status") == "success"

    text = str(result).strip()

    text = text.lower()
    if not text or text.startswith(("error", "warning")):
        return False
    return text.startswith("success") or "already running" in text


def _format_tool_result(result: object) -> str:
    if isinstance(result, (dict, list)):
        return json.dumps(result, sort_keys=True)
    return str(result)


def _tool_message(tool_call: dict, content: any, index: int = 0) -> ToolMessage:
    tool_call_id = str(tool_call.get("id") or f"pipeline_call_{index}")
    return ToolMessage(content=content, tool_call_id=tool_call_id)


def _failed_update(
    state: AgentState,
    tool_messages: list[ToolMessage],
    reason: str,
) -> dict:
    print(f"[PIPELINE FAILED]: {reason}")
    return {
        "messages": state["messages"] + tool_messages,
        "iterations": state.get("iterations", 0),
        "terminated": True,
        "outcome": "failure",
        "failure_reason": reason,
    }


# ── 3. GRAPH NODES ──────────────────────────────────────────────────────────
def _compact_ledger_entry(tool_name: str, arguments: dict, result: object) -> str:
    """Persist only facts needed after older tool turns are removed."""
    if tool_name == "execute_pick_script" and isinstance(result, dict):
        if result.get("success") is True:
            return f"Pick succeeded; holding {result.get('object_id', arguments.get('object_id', 'object'))}."
        return f"Pick failed safely: {str(result.get('reason', 'unknown reason'))[:180]}"
    if tool_name == "execute_place_script" and isinstance(result, dict):
        if result.get("success") is True:
            return f"Place succeeded at {result.get('destination_object_id', arguments.get('destination_object_id', 'destination'))}."
        return f"Place failed safely: {str(result.get('reason', 'unknown reason'))[:180]}"
    if tool_name == "move_rail_to_object":
        if isinstance(result, dict) and result.get("success") is not True:
            return f"Rail/look-at failed safely: {str(result.get('reason', 'unknown reason'))[:180]}"
        return f"Rail/look-at completed for {arguments.get('object_id', 'requested object')}."
    if tool_name == "scan_object" and isinstance(result, dict):
        return (
            f"Object scan for {arguments.get('object_id', 'object')}: "
            f"{result.get('viewpoints_completed', 0)}/"
            f"{result.get('viewpoints_planned', '?')} viewpoints; "
            f"final_position={result.get('final_position', 'unavailable')}; "
            f"result={result.get('status', 'unknown')}."
        )
    if tool_name == "get_camera_frame":
        if not isinstance(result, dict):
            return "Isolated visual verification did not return structured evidence."
        return (
            "Isolated visual verification: "
            f"target_confirmed={str(result.get('target_confirmed') is True).lower()}, "
            f"safe_for_action={str(result.get('safe_for_action') is True).lower()}. "
            f"Evidence: {str(result.get('observation', {}).get('visible_evidence', ''))[:180]}"
        )
    if tool_name == "start_joint_controller":
        return "Robot startup posture completed and gripper-open command issued."
    if tool_name == "search_semantic_objects" and isinstance(result, dict):
        matches = result.get("objects", [])
        facts: list[str] = []
        for item in matches[:5]:
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("object_id", "unknown"))
            class_name = str(item.get("class_name", "unknown"))
            state = str(item.get("state", "unknown"))
            confidence = item.get("confidence", "?")
            similarity = item.get("cosine_similarity", "?")
            x, y, z = item.get("x", "?"), item.get("y", "?"), item.get("z", "?")
            facts.append(
                f"{object_id} ({class_name}, {state}, confidence={confidence}, "
                f"text_similarity={similarity}, xyz=({x}, {y}, {z}))"
            )
        return (
            f"Semantic search {arguments.get('query', '')!r} returned: "
            f"{'; '.join(facts) if facts else 'no matches'}."
        )
    if tool_name in {"targeted_search", "general_mapping"}:
        return f"{tool_name} result: {str(result)[:180]}"
    return f"{tool_name}: {str(result)[:180]}"


def _semantic_query_key(query: object) -> str:
    """Normalize a query solely for duplicate-query suppression."""
    return " ".join(str(query).casefold().split())


def _messages_for_llm(state: AgentState) -> list[BaseMessage]:
    """Bound prompt size while retaining task, durable state, and latest tool turn."""
    messages = state["messages"]
    if not messages:
        return []

    # Retain the initial system/task messages; historical tool exchanges are
    # represented by the ledger below rather than repeatedly consuming context.
    prompt: list[BaseMessage] = [messages[0]]
    if len(messages) > 1:
        prompt.append(messages[1])

    ledger = state.get("execution_ledger", [])[-8:]
    if ledger:
        prompt.append(HumanMessage(content=(
            "Execution ledger (durable facts from earlier steps):\n- "
            + "\n- ".join(ledger)
        )))

    # A tool response must remain paired with the immediately preceding AI
    # tool-call message. Keep that one current turn, including a current image.
    latest_ai_index = max(
        (index for index, message in enumerate(messages) if isinstance(message, AIMessage)),
        default=-1,
    )
    if latest_ai_index >= 0:
        prompt.extend(messages[latest_ai_index:])
    return prompt


def call_llm(state: AgentState) -> dict:
    print("\n[LLM IS EVALUATING TASK...]")
    response = llm.invoke(_messages_for_llm(state))

    if getattr(response, "tool_calls", None):
        pass
        #print(f" -> [DEBUG] NATIVE TOOL CALL DETECTED: {response.tool_calls}")
    else:
        print(" -> [DEBUG] NO TOOLS DETECTED IN AI RESPONSE.")

    return {
        "messages": state["messages"] + [response],
        "iterations": state.get("iterations", 0) + 1,
    }


def execute_tools(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_calls = list(getattr(last_message, "tool_calls", None) or [])
    if len(tool_calls) != 1:
        reason = (
            "Exactly one high-level tool call is required per planner response; "
            f"received {len(tool_calls)}."
        )
        error_result = json.dumps({"status": "failure", "reason": reason})
        messages = [
            _tool_message(tool_call, error_result, index)
            for index, tool_call in enumerate(tool_calls)
        ]
        return _failed_update(state, messages, reason)

    tool_call = tool_calls[0]
    tool_name = str(tool_call.get("name") or "")
    arguments = tool_call.get("args")
    if not isinstance(arguments, dict):
        reason = f"Tool {tool_name!r} arguments must be a JSON object."
        return _failed_update(
            state,
            [_tool_message(tool_call, json.dumps({"status": "failure", "reason": reason}))],
            reason,
        )

    validation_error = _validate_tool_for_stage(state, tool_name, arguments)
    if validation_error is not None:
        return _failed_update(
            state,
            [
                _tool_message(
                    tool_call,
                    json.dumps({"status": "failure", "reason": validation_error}),
                )
            ],
            validation_error,
        )

    try:
        import sys
        import re
        if "--show-think" in sys.argv:
            thought = arguments.get("thought", "")
            if thought:
                # Optionally strip the actual tags if they were included
                thought = re.sub(r'</?think>', '', thought).strip()
                print(f"\n[AI THOUGHT]:\n{thought}")
        
        print(f"\n[TOOL TRIGGERED]: Executing function {tool_name!r}...")
        query_key = _semantic_query_key(arguments.get("query", ""))
        prior_search = state.get("semantic_query_cache", {}).get(query_key)
        if tool_name == "search_semantic_objects" and prior_search:
            raw_result = {
                "status": "success",
                "query": arguments.get("query", ""),
                "already_queried": True,
                "reason": (
                    "This query was already run against the current semantic map. "
                    "Use the durable execution ledger result and take the next planning action; "
                    "repeat it only after targeted_search or general_mapping."
                ),
                "objects": [],
            }
        else:
            raw_result = tools_impl[tool_name](arguments)
        print(f"[TOOL COMPLETED]: {tool_name!r} execution completed.")
    except Exception as exc:
        reason = f"{tool_name} raised an exception: {exc}"
        return _failed_update(
            state,
            [_tool_message(tool_call, json.dumps({"status": "failure", "reason": reason}))],
            reason,
        )

    result_message = _tool_message(tool_call, _format_tool_result(raw_result))
    recoverable_targeted_search_failure = (
        tool_name == "targeted_search"
        and isinstance(raw_result, dict)
        and raw_result.get("status") == "failure"
        and raw_result.get("state") == "couldnt_find"
    )
    recoverable_manipulation_failure = (
        tool_name in {"execute_pick_script", "execute_place_script"}
        and isinstance(raw_result, dict)
        and raw_result.get("status") == "failure"
    )
    recoverable_motion_failure = (
        isinstance(raw_result, dict)
        and raw_result.get("status") == "failure"
        and raw_result.get("recoverable") is True
    )
    if (
        not _tool_result_succeeded(tool_name, raw_result)
        and not recoverable_targeted_search_failure
        and not recoverable_manipulation_failure
        and not recoverable_motion_failure
    ):
        reason = f"{tool_name} failed: {_format_tool_result(raw_result)}"
        return _failed_update(state, [result_message], reason)

    located_target = ""
    if tool_name == "targeted_search":
        requested_target = _normalized_argument(arguments, "target_object")
        located_target = _normalized_argument(raw_result, "target")
        if not located_target or located_target != requested_target:
            reason = (
                "Search returned an invalid target: requested "
                f"{requested_target!r}, received {located_target!r}."
            )
            return _failed_update(state, [result_message], reason)

    called_identifier = tool_name
    if tool_name == "move_rail_to_object":
        target = _normalized_argument(arguments, "object_id")
        if target == "home":
            called_identifier = "move_rail_to_object_home"

    tools_called = state.get("tools_called", []) + [called_identifier]
    execution_ledger = (
        state.get("execution_ledger", [])
        + [_compact_ledger_entry(tool_name, arguments, raw_result)]
    )[-8:]
    semantic_query_cache = dict(state.get("semantic_query_cache", {}))
    last_visual_verification = state.get("last_visual_verification", {})
    last_navigation_object_id = state.get("last_navigation_object_id", "")
    if tool_name == "get_camera_frame":
        last_visual_verification = raw_result if isinstance(raw_result, dict) else {}
    if tool_name == "move_rail_to_object":
        if isinstance(raw_result, dict) and raw_result.get("success") is True:
            last_navigation_object_id = str(arguments.get("object_id") or "").strip()
        else:
            last_navigation_object_id = ""
    elif tool_name in {
        "start_joint_controller", "targeted_search", "general_mapping",
        "move_rail_relative", "turn_panda_arm", "home_panda_arm",
    }:
        last_navigation_object_id = ""
    if tool_name in {"targeted_search", "general_mapping", "scan_object"}:
        # These motions can add, update, or expire map tracks.
        semantic_query_cache.clear()
    elif tool_name == "search_semantic_objects" and not raw_result.get("already_queried", False):
        semantic_query_cache[_semantic_query_key(arguments.get("query", ""))] = execution_ledger[-1]

    if recoverable_targeted_search_failure or recoverable_motion_failure:
        print(
            "[PIPELINE] Motion/search failure recovered safely; "
            "returning result to the VLM for replanning."
        )

    update = {
        "messages": state["messages"] + [
            result_message,
        ],
        "iterations": state.get("iterations", 0),
        "terminated": False,
        "outcome": "in_progress",
        "failure_reason": "",
        "tools_called": tools_called,
        "execution_ledger": execution_ledger,
        "semantic_query_cache": semantic_query_cache,
        "last_visual_verification": last_visual_verification,
        "last_navigation_object_id": last_navigation_object_id,
    }
    if tool_name == "finish_task":
        summary = str(arguments["summary"])
        print(f"\n[PIPELINE COMPLETE]: {summary}")
        update.update(
            {
                "terminated": True,
                "outcome": "success",
            }
        )
    return update


def should_continue(state: AgentState) -> str:
    if state.get("terminated", False):
        return "end"
    if state.get("iterations", 0) >= 20:
        print("\n[WARNING]: Maximum iterations reached. Forcing end.")
        return "end"
    last_message = state["messages"][-1]
    return "execute_tools" if getattr(last_message, "tool_calls", None) else "end"


def route_after_tools(state: AgentState) -> str:
    return "end" if state.get("terminated", False) else "call_llm"


# ── 4. BUILD GRAPH SCHEMA ──────────────────────────────────────────────────
workflow = StateGraph(AgentState)
workflow.add_node("call_llm", call_llm)
workflow.add_node("execute_tools", execute_tools)
workflow.add_edge(START, "call_llm")
workflow.add_conditional_edges(
    "call_llm",
    should_continue,
    {"execute_tools": "execute_tools", "end": END},
)
workflow.add_conditional_edges(
    "execute_tools",
    route_after_tools,
    {"call_llm": "call_llm", "end": END},
)
agent = workflow.compile()
