import base64
import operator
from typing import Literal, Annotated, TypedDict
from langchain_ollama import ChatOllama
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langgraph.graph import END
from langgraph.prebuilt import ToolNode

from tools import get_image_ros2, publish_joint_command
from ui import console
from rich.panel import Panel

# --- MODEL SETUP ---
tools = [publish_joint_command]
model_text = ChatOllama(model="qwen3-vl:8b", temperature=0.0).bind_tools(tools)
model_vlm = ChatOllama(model="qwen3-vl:8b", temperature=1.0)

# --- STATE DEFINITION ---
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    model_calls: int

# --- UTILITIES ---
def prompt_func(data):
    """Formats image and text into a multimodal message."""
    text = data["text"]
    image_path = data["image"]

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{encoded_image}",
    }
    text_part = {"type": "text", "text": text}

    return [HumanMessage(content=[image_part, text_part])]

def _get_latest_text_task(messages) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            return msg.content
    return ""

# --- GRAPH NODES ---
def analyzer_node(state: MessagesState):
    """Text-only fast tool reasoning."""
    task_text = _get_latest_text_task(state["messages"])
    response = model_text.invoke([
        SystemMessage(content="You are a robotic tool router. If the task requires moving the robot base, use the publish_joint_command tool immediately. Towards the wall requires positive values. Otherwise, just output 'NO_TOOL'."),
        HumanMessage(content=task_text)
    ])
    return {"messages": [response]}

def vlm_node(state: MessagesState):
    """Image-only reasoning."""
    image_msg = state["messages"][-1]
    response = model_vlm.invoke([
        SystemMessage(
            content="""
            You are a robotic task orchestrator. 
            For a given image and a task, you should give subtasks with format "pick up the <object-name> and place it into the <object-name>".
            If the task is asking a question to you, answer it.
            """
        ),
        image_msg
    ])
    return {"messages": [response], "model_calls": state.get('model_calls', 0) + 1}

def get_image_node(state: MessagesState):
    """Captures a frame from ROS 2 and updates the state."""
    task = _get_latest_text_task(state["messages"])
    res = get_image_ros2(task=task)
    
    if isinstance(res, str) and res.startswith("Error"):
        return {"messages": [HumanMessage(content=f"Failed to capture image: {res}")]}
    
    return {"messages": prompt_func(res)}

_tool_node = ToolNode(tools)

def tool_node(state: MessagesState):
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            console.print(Panel(
                f"[bold cyan]I used the tool[/bold cyan] [bold magenta]{tool_name}[/bold magenta]!",
                title="[bold yellow]🛠️  Tool Execution[/bold yellow]",
                border_style="yellow",
                expand=False
            ))
    return _tool_node.invoke(state)

def should_use_tool_or_image(state: MessagesState) -> Literal["tool_node", "get_image"]:
    """Determines if the tool should be executed or if we should get an image."""
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool_node"
    return "get_image"
