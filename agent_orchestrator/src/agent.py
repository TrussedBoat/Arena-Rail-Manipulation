from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage

# Local imports
from ui import UI
from nodes import MessagesState, analyzer_node, get_image_node, tool_node, vlm_node, should_use_tool_or_image

import time

# --- MEMORY CONFIGURATION ---
checkpointer = InMemorySaver()

def build_agent():
    """Constructs and compiles the LangGraph agent."""
    builder = StateGraph(MessagesState)

    # Add nodes
    builder.add_node("analyzer", analyzer_node)
    builder.add_node("tool_node", tool_node)
    builder.add_node("get_image", get_image_node)
    builder.add_node("vlm_node", vlm_node)

    # Add edges
    builder.add_edge(START, "analyzer")
    builder.add_conditional_edges(
        "analyzer",
        should_use_tool_or_image,
        {
            "tool_node": "tool_node",
            "get_image": "get_image"
        }
    )
    builder.add_edge("tool_node", END)
    builder.add_edge("get_image", "vlm_node")
    builder.add_edge("vlm_node", END)

    return builder.compile(checkpointer=checkpointer)

def main():
    agent = build_agent()
    UI.print_header()

    try:
        while True:
            task = UI.get_input("Which task should I complete? (q to quit)")
            if task.lower() == "q":
                break

            with UI.show_status("Analyzing the task and environment..."):
                initial_state = {"messages": [HumanMessage(content=task)], "model_calls": 0}
                # Use a fixed thread_id for local session memory
                config = {"configurable": {"thread_id": "1"}}
                
                start_time = time.time()
                result = agent.invoke(initial_state, config=config)
                elapsed_time = time.time() - start_time

            UI.display_result(result["messages"], elapsed_time=elapsed_time)

    except KeyboardInterrupt:
        pass

    UI.print_goodbye()

if __name__ == "__main__":
    main()
