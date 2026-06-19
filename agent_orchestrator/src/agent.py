import json
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage

from tools import (
    start_joint_controller,
    get_latest_ros_image,
    get_current_joint_states,
    move_rail_to_object, 
    home_panda_arm,
    turn_panda_arm,
    execute_pick_script,
    execute_place_script,
)

MODEL_NAME = "Qwen3.6-35B"

tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "start_joint_controller",
            "description": "Initializes the global ROS 2 joint controller. This MUST be called at the very beginning of the task before attempting to move the rails or arm.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_robot_joint_states",
            "description": "Read the live metrics of the hardware. Use this to check if the arm is at 0.0 before moving rails.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "home_panda_arm",
            "description": "Command the panda_joint1 actuator to return to its default safe home position (0.0 radians). Use this if the arm is displaced before rail operations.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_latest_image_from_ros",
            "description": "Get visual workspace data from the camera stream. Use this if you need to see the scene.",
            "parameters": {
                "type": "object",
                "properties": {"timeout_sec": {"type": "integer"}},
                "required": ["timeout_sec"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_rail_to_object",
            "description": "Moves the robot base along the rails to align with a specific object in the room.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_object": {
                        "type": "string", 
                        "description": "The name of the object to move to (e.g., 'pan', 'plate')."
                    }
                },
                "required": ["target_object"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "turn_panda_arm",
            "description": "Turns the robot arm (panda_joint1) to face a specific direction. Used after moving the rails to face the object.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_rad": {
                        "type": "number", 
                        "description": "The target angle in radians (e.g., 1.57 for Left, -1.57 for Right)."
                    }
                },
                "required": ["target_rad"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_pick_script",
            "description": "Runs the hardware shell script to physically pick up the object.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_object": {
                        "type": "string",
                        "description": "The name of the object to pick up (e.g., 'lemon')."
                    }
                },
                "required": ["target_object"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_place_script",
            "description": "Runs the hardware shell script to physically place the object. Call this ONLY after grasping the object, moving to the destination, pointing the arm, and visually confirming the location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_object": {
                        "type": "string",
                        "description": "The name of the destination to place the object into (e.g., 'plate')."
                    }
                },
                "required": ["target_object"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish_task",
            "description": "Call this tool ONLY when the entire task (Pick, Place, and Homing) is 100% complete. This tells the system to shut down the pipeline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A brief 1-sentence summary of what was successfully accomplished."
                    }
                },
                "required": ["summary"]
            }
        }
    }
]

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",
    model=MODEL_NAME,
    temperature=0.2,
    timeout=120.0,
    max_tokens=4096
).bind_tools(tool_definitions)

class AgentState(TypedDict):
    messages: list[BaseMessage]
    iterations: int

tools_impl = {
    "start_joint_controller": lambda a: start_joint_controller(),
    "check_robot_joint_states": lambda a: json.dumps(get_current_joint_states()),
    "home_panda_arm": lambda a: home_panda_arm(),
    "get_latest_image_from_ros": lambda a: get_latest_ros_image(a.get("timeout_sec", 10)),
    "move_rail_to_object": lambda a: move_rail_to_object(a.get("target_object")),
    "turn_panda_arm": lambda a: turn_panda_arm(a.get("target_rad")),
    "execute_pick_script": lambda a: execute_pick_script(a.get("target_object")),
    "execute_place_script": lambda a: execute_place_script(a.get("target_object")),
}

# ── 3. GRAPH NODES ──────────────────────────────────────────────────────────
def call_llm(state: AgentState) -> dict:
    print("\n[LLM IS EVALUATING TASK...]")
    response = llm.invoke(state["messages"])
    
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f" -> [DEBUG] NATIVE TOOL CALL DETECTED: {response.tool_calls}")
    else:
        print(" -> [DEBUG] NO TOOLS DETECTED IN AI RESPONSE.")
        
    return {"messages": state["messages"] + [response], "iterations": state.get("iterations", 0) + 1}

def execute_tools(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    new_tool_messages = []
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tc in last_message.tool_calls:
            try:
                print(f"\n[TOOL TRIGGERED]: Executing function '{tc['name']}'...")
                raw_result = tools_impl[tc["name"]](tc["args"])
                print(f"[TOOL COMPLETED]: '{tc['name']}' execution completed.")
                
                if tc["name"] == "get_latest_image_from_ros":
                    new_tool_messages.append(
                        HumanMessage(content=[
                            {"type": "text", "text": "Image captured successfully."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{raw_result}"}}
                        ])
                    )
                else:
                    new_tool_messages.append(HumanMessage(content=f"Tool Execution Result: {raw_result}"))
            except Exception as e:
                print(f"[TOOL ERROR]: '{tc['name']}' execution failure: {e}")
                new_tool_messages.append(HumanMessage(content=f"error executing tool {tc['name']}: {e}"))
                
    return {"messages": state["messages"] + new_tool_messages, "iterations": state.get("iterations", 0)}

def should_continue(state: AgentState) -> str:
    # 1. Safety Limit Fallback
    if state.get("iterations", 0) >= 20: 
        print("\n[WARNING]: Maximum iterations reached. Forcing end.")
        return "end"
        
    last_message = state["messages"][-1]
    
    # 2. Tool Routing & Interception
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        
        # Check if the LLM wants to end the task
        for tool in last_message.tool_calls:
            if tool["name"] == "finish_task":
                # Extract the summary argument the LLM generated
                summary = tool["args"].get("summary", "Task complete.")
                print(f"\n[PIPELINE COMPLETE]: {summary}")
                print("[SYSTEM]: 'finish_task' tool intercepted. Terminating graph loop gracefully.")
                return "end" # <-- Short-circuits the graph instantly!
                
        # If it's any other tool, route normally
        return "execute_tools"
        
    # 3. If no tools are called and we are here, end naturally
    return "end"

# ── 4. BUILD GRAPH SCHEMA ──────────────────────────────────────────────────
workflow = StateGraph(AgentState)
workflow.add_node("call_llm", call_llm)
workflow.add_node("execute_tools", execute_tools)
workflow.add_edge(START, "call_llm")
workflow.add_conditional_edges("call_llm", should_continue, {"execute_tools": "execute_tools", "end": END})
workflow.add_edge("execute_tools", "call_llm")
agent = workflow.compile()
