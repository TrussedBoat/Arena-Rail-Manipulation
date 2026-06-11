#!/usr/bin/env python3
import operator
import json
import base64
import time
from typing import TypedDict, Annotated, Sequence

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

# ── 1. ROS 2 Image Capture Helper ───────────────────────────────────────────
class SingleImageSubscriber(Node):
    def __init__(self):
        super().__init__('agent_image_subscriber')
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.listener_callback, 10
        )
        self.bridge = CvBridge()
        self.latest_b64_image = None

    def listener_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            _, buffer = cv2.imencode('.png', cv_img)
            self.latest_b64_image = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            self.get_logger().error(f"Failed to process image message: {e}")

def get_latest_ros_image(timeout_sec=5.0) -> str:
    node = SingleImageSubscriber()
    start = time.time()
    
    print("Waiting for image from ROS 2 topic /camera/image_raw...")
    while node.latest_b64_image is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        if (time.time() - start) > timeout_sec:
            node.destroy_node()
            raise TimeoutError("Timed out waiting for ROS 2 image message.")
            
    img_data = node.latest_b64_image
    node.destroy_node()
    return img_data

# ── 2. LangGraph & LLM Configuration ────────────────────────────────────────
MODEL_NAME = "Qwen3.6-35B"

tool_definitions = [{
    "type": "function",
    "function": {
        "name": "get_latest_image_from_ros",
        "description": "Call this tool if you need visual data from the workspace camera to complete or plan the task.",
        "parameters": {
            "type": "object",
            "properties": {
                "timeout_sec": {"type": "integer", "description": "Timeout in seconds to wait for an image"}
            },
            "required": ["timeout_sec"]
        }
    }
}]

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",
    model=MODEL_NAME,
    temperature=0.2,
    timeout=120.0,
    max_tokens=4096
).bind_tools(tool_definitions)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    iterations: int

# Modified execution: execute_tool returns raw data
tools_impl = {
    "get_latest_image_from_ros": lambda a: get_latest_ros_image(a.get("timeout_sec", 10)),
}

def execute_tool(tool_name: str, args: dict) -> str:
    fn = tools_impl.get(tool_name)
    if fn is None:
        return f"Unknown tool: {tool_name}"
    return fn(args)

def call_llm(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response], "iterations": state.get("iterations", 0) + 1}

# Modified to structural multimodal payload injection
def execute_tools(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        new_messages = []
        for tc in last_message.tool_calls:
            try:
                # ── Tool Start Notification ──
                print(f"\n[TOOL TRIGGERED]: Executing function '{tc['name']}'...")
                
                raw_result = execute_tool(tc["name"], tc["args"])
                
                # ── Tool Success Notification ──
                print(f"[TOOL COMPLETED]: '{tc['name']}' successfully fetched data.")
                
                if tc["name"] == "get_latest_image_from_ros":
                    new_messages.append(
                        HumanMessage(content=[
                            {"type": "text", "text": "Here is the requested image from the live ROS 2 stream. Proceed with task planning."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{raw_result}"}}
                        ])
                    )
                else:
                    new_messages.append(HumanMessage(content=f"Tool result: {raw_result}"))
            except Exception as e:
                print(f"[TOOL ERROR]: '{tc['name']}' failed: {e}")
                new_messages.append(HumanMessage(content=f"error executing tool {tc['name']}: {e}"))
                
        return {
            "messages": new_messages,
            "iterations": state.get("iterations", 0)
        }
    return {"messages": [], "iterations": state.get("iterations", 0)}

def should_continue(state: AgentState) -> str:
    if state.get("iterations", 0) >= 3:
        return "end"
    for msg in reversed(state["messages"]):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            return "execute_tools"
    return "end"

# ── 3. Build Workflow Graph ──────────────────────────────────────────────────
workflow = StateGraph(AgentState)
workflow.add_node("call_llm", call_llm)
workflow.add_node("execute_tools", execute_tools)
workflow.add_edge(START, "call_llm")
workflow.add_conditional_edges("call_llm", should_continue, {
    "execute_tools": "execute_tools",
    "end": END
})
workflow.add_edge("execute_tools", "call_llm")
agent = workflow.compile()

# ── 4. Main Runtime Execution ────────────────────────────────────────────────
import re

# ── UPDATED TRACE PRINTING FOR main() ─────────────────────────────────────
def main():
    rclpy.init()
    try:
        input_state = {
            "messages": [
                SystemMessage(content=(
                    "You are a robotic task orchestrator. When assigned a manipulation task, "
                    "if you require sensory feedback to see the scene state, call your "
                    "get_latest_image_from_ros tool first. "
                    "Once you have the image, output subtasks using format: "
                    "'pick up the <object-name> and place it into the <object-name>.'"
                )),
                HumanMessage(content="task: arrange the table")
            ],
            "iterations": 0
        }
        
        print("Starting conditional multi-modal agent pipeline...")
        start_time = time.time()
        result = agent.invoke(input_state)
        elapsed_time = time.time() - start_time
        
        print("\n--- WORKFLOW EXECUTION TRACE ---")
        seen_responses = set()
        
        for msg in result["messages"]:
            if not msg.content:
                continue
                
            # 1. Handle Multimodal Payload Display
            if isinstance(msg.content, list):
                print(f"\n[{msg.type.upper()}]:")
                print(f"[Multimodal Payload: {msg.content[0]['text']}]")
                continue

            # 2. Clean up AI Reasoning tags (<think>...</think>) if present
            cleaned_content = msg.content
            if msg.type == "ai":
                # Remove any internal thinking blocks from the text output
                cleaned_content = re.sub(r'<think>.*?</think>', '', cleaned_content, flags=re.DOTALL)
                # Fallback: if server formats reasoning separated by double newlines
                if "\n\n" in cleaned_content and len(cleaned_content.split("\n\n")) > 1:
                    cleaned_content = cleaned_content.split("\n\n")[-1]
            
            cleaned_content = cleaned_content.strip()
            if not cleaned_content:
                continue

            # 3. Prevent duplicate blocks from printing
            if cleaned_content in seen_responses:
                continue
            seen_responses.add(cleaned_content)
            
            print(f"\n[{msg.type.upper()}]:")
            print(cleaned_content)
                
        print(f"\n--- METRICS ---\nTotal Execution Time: {elapsed_time:.2f} seconds")

    except Exception as e:
        print(f"Pipeline Error: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()