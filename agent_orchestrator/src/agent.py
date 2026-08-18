import base64
import binascii
import json
from typing import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
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
    get_settled_vlm_image,
    get_semantic_objects,
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
                        "description": "Exact object_id returned by get_semantic_objects.",
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
            "name": "get_semantic_objects",
            "description": "Semantic planning lookup. Query every relevant canonical base class before choosing search or navigation. Returns only confirmed matching objects with IDs, confidence, global XYZ, and last_seen; classes_without_match means no usable map entry. It is map evidence, not visual proof.",
            "parameters": {
                "type": "object",
                "properties": {
                    "class_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "Relevant canonical base detector classes, for example [\"apple\", \"bowl\"]. Do not include visual qualifiers such as color.",
                    }
                },
                "required": ["class_names"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_rail_to_object",
            "description": "Rail-align and point at one selected confirmed semantic object ID. Use the exact ID returned by get_semantic_objects; duplicate classes must be disambiguated by ID.",
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
                        "description": "Exact confirmed destination object_id returned by get_semantic_objects.",
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
            "description": "Wait two seconds for the robot/camera to settle, capture two new ROS wrist-camera frames, and return the later frame. Call this immediately before every execute_pick_script or execute_place_script and visually verify the needed object.",
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
    "get_camera_frame": lambda _: get_settled_vlm_image(
        timeout_sec=10.0, settle_sec=2.0, frame_count=2
    ),
    "get_semantic_objects": lambda args: get_semantic_objects(args.get("class_names")),
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

    if tool_name in ("execute_pick_script", "execute_place_script"):
        if not tools_called or tools_called[-1] != "get_camera_frame":
            return (
                "Immediately before every pick or place hardware execution, call "
                "get_camera_frame and visually verify the required object."
            )

    return None


def _is_jpeg_base64(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        image_bytes = base64.b64decode(value.strip(), validate=True)
    except (ValueError, binascii.Error):
        return False
    # JPEG start-of-image marker plus a valid marker prefix.
    return image_bytes.startswith(b"\xff\xd8\xff")


def _tool_result_succeeded(tool_name: str, result: object) -> bool:
    if isinstance(result, dict):
        if tool_name in ("targeted_search", "general_mapping"):
            return result.get("status") == "success" and result.get("success") is True
        if "success" in result:
            return result.get("success") is True
        return result.get("status") == "success"

    text = str(result).strip()
    if tool_name == "get_camera_frame":
        return _is_jpeg_base64(text)

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
        import sys
        import re
        if "--show-think" in sys.argv:
            thought = arguments.get("thought", "")
            if thought:
                # Optionally strip the actual tags if they were included
                thought = re.sub(r'</?think>', '', thought).strip()
                print(f"\n[AI THOUGHT]:\n{thought}")
        
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

    camera_frame_message = None
    if tool_name == "get_camera_frame" and _is_jpeg_base64(raw_result):
        # llama.cpp VLMs reliably parse image_url blocks in a user message, but
        # may render image data in a tool result as plain text. Keep the tool
        # protocol response text-only and provide the JPEG through image_url.
        result_message = _tool_message(tool_call, "Camera frame captured.")
        camera_frame_message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Use this current wrist-camera frame to visually verify the object.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{raw_result}"},
                },
            ]
        )
    else:
        result_message = _tool_message(tool_call, _format_tool_result(raw_result))
    recoverable_targeted_search_failure = (
        tool_name == "targeted_search"
        and isinstance(raw_result, dict)
        and raw_result.get("status") == "failure"
        and raw_result.get("state") == "couldnt_find"
    )
    if not _tool_result_succeeded(tool_name, raw_result) and not recoverable_targeted_search_failure:
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

    if recoverable_targeted_search_failure:
        print(
            "[PIPELINE] Targeted search did not find the target; "
            "returning result to the VLM for replanning."
        )

    update = {
        "messages": state["messages"] + [
            result_message,
            *([camera_frame_message] if camera_frame_message is not None else []),
        ],
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
