import json
from typing import TypedDict

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from config import get_runtime_config
from tools import (
    execute_pick_script,
    execute_place_script,
    move_rail_to_object,
    general_mapping,
    targeted_search,
    start_joint_controller,
    turn_panda_arm,
    home_panda_arm,
    move_rail_relative,
    move_eef_to_pose,
)

runtime_config = get_runtime_config()
MODEL_NAME = runtime_config.vlm.model_alias

tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "start_joint_controller",
            "description": (
                "Initialize the robot, move rail_j1 to -1.1m, set the arm to its "
                "safe startup pose, and open the gripper. Always call this first."
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
            "name": "targeted_search",
            "description": (
                "Human-like targeted search: pans the wrist camera right then left to find "
                "a specific object on the tables. Use this as the primary search tool when "
                "looking for an object. If the object is not found from the current position, "
                "the robot moves to the table centre and tries once more. "
                "Pass one normalized canonical label such as 'apple'."
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
                "Full rail sweep search: moves the robot along the entire rail while scanning "
                "with YOLO to map object locations. Use this as a fallback if targeted_search "
                "fails. Pass one normalized canonical label such as 'apple'."
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
            "name": "execute_pick_script",
            "description": "Execute the physical picking motion to grab the target object. Only use this if the user explicitly asked to pick something up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_object": {
                        "type": "string",
                        "description": "The normalized target label.",
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
            "name": "move_rail_to_object",
            "description": "Move the robot's base rail to the location of a known object (e.g. 'apple', 'purple bowl', or 'home'). Use this to approach an object.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_object": {
                        "type": "string",
                        "description": "Destination label, such as 'apple', 'purple bowl' or 'home'.",
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
            "description": "Execute the physical placing motion to drop a held object at the current location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_object": {
                        "type": "string",
                        "description": "The placement destination label.",
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
            "name": "move_eef_to_pose",
            "description": (
                "Plan and execute a self-collision-free RRT motion for the Franka end "
                "effector to an absolute pose relative to panda_link0. XYZ values are "
                "metres and roll/pitch/yaw are radians. The plan is rejected if it violates "
                "joint limits or the configured singularity threshold."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "EEF X position in metres."},
                    "y": {"type": "number", "description": "EEF Y position in metres."},
                    "z": {"type": "number", "description": "EEF Z position in metres."},
                    "roll": {"type": "number", "description": "Roll in radians."},
                    "pitch": {"type": "number", "description": "Pitch in radians."},
                    "yaw": {"type": "number", "description": "Yaw in radians."},
                },
                "required": ["x", "y", "z", "roll", "pitch", "yaw"],
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
BOWL_TARGETS = {"bowl", "purple bowl"}


tools_impl = {
    "start_joint_controller": lambda _: start_joint_controller(),
    "general_mapping": lambda args: general_mapping(
        args.get("target_object")
    ),
    "targeted_search": lambda args: targeted_search(
        args.get("target_object")
    ),
    "execute_pick_script": lambda args: execute_pick_script(
        args.get("target_object")
    ),
    "move_rail_to_object": lambda args: move_rail_to_object(
        args.get("target_object")
    ),
    "move_rail_relative": lambda args: move_rail_relative(
        args.get("relative_distance_m")
    ),
    "execute_place_script": lambda args: execute_place_script(
        args.get("target_object")
    ),
    "turn_panda_arm": lambda args: turn_panda_arm(
        args.get("target_rad")
    ),
    "home_panda_arm": lambda _: home_panda_arm(),
    "move_eef_to_pose": lambda args: move_eef_to_pose(
        args.get("x"),
        args.get("y"),
        args.get("z"),
        args.get("roll"),
        args.get("pitch"),
        args.get("yaw"),
    ),
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
        if not tools_called or tools_called[-1] != "move_rail_to_object_home":
            return "You must return to the 'home' position using 'move_rail_to_object' with target_object 'home' before finishing the task."
        if not str(arguments.get("summary") or "").strip():
            return "finish_task requires a non-empty summary."

    target = _normalized_argument(arguments, "target_object")
    if tool_name == "search_and_locate_with_yolo" and not target:
        return "Search requires a non-empty canonical pickup target."

    return None


def _tool_result_succeeded(tool_name: str, result: object) -> bool:
    if isinstance(result, dict):
        if tool_name == "search_and_locate_with_yolo":
            return result.get("status") == "success" and result.get("success") is True
        if "success" in result:
            return result.get("success") is True
        return result.get("status") == "success"

    text = str(result).strip().lower()
    if not text or text.startswith(("error", "warning")):
        return False
    return text.startswith("success") or "already running" in text


def _format_tool_result(result: object) -> str:
    if isinstance(result, (dict, list)):
        return json.dumps(result, sort_keys=True)
    return str(result)


def _tool_message(tool_call: dict, content: str, index: int = 0) -> ToolMessage:
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
def call_llm(state: AgentState) -> dict:
    print("\n[LLM IS EVALUATING TASK...]")
    response = llm.invoke(state["messages"])

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
        print(f"\n[TOOL TRIGGERED]: Executing function {tool_name!r}...")
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
    if not _tool_result_succeeded(tool_name, raw_result):
        reason = f"{tool_name} failed: {_format_tool_result(raw_result)}"
        return _failed_update(state, [result_message], reason)

    located_target = ""
    if tool_name in ("targeted_search", "general_mapping"):
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
        target = _normalized_argument(arguments, "target_object")
        if target == "home":
            called_identifier = "move_rail_to_object_home"

    tools_called = state.get("tools_called", []) + [called_identifier]

    update = {
        "messages": state["messages"] + [result_message],
        "iterations": state.get("iterations", 0),
        "terminated": False,
        "outcome": "in_progress",
        "failure_reason": "",
        "tools_called": tools_called,
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
