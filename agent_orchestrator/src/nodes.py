import base64
import operator
from typing import Literal, Annotated, TypedDict
from langchain_ollama import ChatOllama
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langgraph.graph import END

from tools import get_image_ros2

# --- MODEL SETUP ---
model = ChatOllama(model="qwen3-vl:8b-instruct", temperature=0.5)

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

# --- GRAPH NODES ---
def model_call(state: MessagesState):
    """LLM decides the next action based on context."""
    return {
        "messages": [
            model.invoke(
                [
                    SystemMessage(
                        content="""
                        You are a robotic task orchestrator. 
                        For a given image and a task, you should give subtasks with format "pick up the <object-name> and place it into the <object-name>".
                        If the task is asking a question to you, answer it.
                        """
                    )
                ]
                + state["messages"]
            )
        ],
        "model_calls": state.get('model_calls', 0) + 1
    }

def get_image_node(state: MessagesState):
    """Captures a frame from ROS 2 and updates the state."""
    task = state["messages"][-1].content
    res = get_image_ros2(task=task)
    
    if isinstance(res, str) and res.startswith("Error"):
        return {"messages": [HumanMessage(content=f"Failed to capture image: {res}")]}
    
    return {"messages": prompt_func(res)}

def tool_node(state: MessagesState):
    """Placeholder for tool execution if tools were bound."""
    # Note: tools are currently disabled in the main flow but the node is kept for structure
    return {"messages": []}

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Determines if the loop should continue."""
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool_node"
    return END