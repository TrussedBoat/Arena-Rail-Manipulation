#!/usr/bin/env python3
import time
import re
import rclpy
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from agent import agent, AgentState

def main():
    rclpy.init()
    try:
        print("Starting interactive dynamic agentic-tool pipeline...")
        print("Type 'quit' or 'exit' at any time to stop the script.\n")
        
        while True:
            user_task = input("\n[USER] Enter your next task: ").strip()
            
            if user_task.lower() in ['quit', 'exit', 'q']:
                print("Shutting down pipeline...")
                break
                
            if not user_task:
                continue
                
            input_state = {
                "messages": [
                    SystemMessage(content=(
                        "You are a strict robotic orchestrator operating via JSON tools.\n"
                        "CRITICAL PIPELINE RULES:\n"
                        "1. STAGE SPLIT: Complete the PICK phase entirely before starting the PLACE phase.\n"
                        "2. VISUAL CHECK: Call 'get_latest_image_from_ros' to check if your target object is in view.\n"
                        "3. NAVIGATION & ORIENTATION: If the target is NOT in view:\n"
                        "   a) Call 'move_rail_to_object'. (This tool automatically ensures arm safety and moves the base).\n"
                        "   b) Call 'turn_panda_arm' using the EXACT angle provided by the previous tool.\n"
                        "   c) Call 'get_latest_image_from_ros' again to confirm the object is now in view.\n"
                        "4. EXECUTION: Once the object is visually confirmed, output 'Action: pick up the <object>' or 'Action: place it into the <object>'."
                    )),
                    HumanMessage(content=f"task: {user_task}")
                ],
                "iterations": 0
            }
            
            print("\n[SYSTEM] Executing task...")
            start_time = time.time()
            result = agent.invoke(input_state)
            elapsed_time = time.time() - start_time
            
            print("\n--- WORKFLOW EXECUTION TRACE ---")
            seen_responses = set()
            
            for msg in result["messages"]:
                if not msg.content:
                    continue
                if isinstance(msg.content, list):
                    print(f"\n[SYSTEM]: [Multimodal Payload Attached]")
                    continue

                cleaned_content = msg.content
                if isinstance(msg, AIMessage):
                    cleaned_content = re.sub(r'<think>.*?</think>', '', cleaned_content, flags=re.DOTALL)
                    if "\n\n" in cleaned_content and len(cleaned_content.split("\n\n")) > 1:
                        cleaned_content = cleaned_content.split("\n\n")[-1]
                
                cleaned_content = cleaned_content.strip()
                if not cleaned_content or cleaned_content in seen_responses:
                    continue
                seen_responses.add(cleaned_content)
                
                print(f"\n[{msg.type.upper()}]:\n{cleaned_content}")
                    
            print(f"\n--- METRICS ---\nTask Execution Time: {elapsed_time:.2f} seconds")
            print("-" * 50) 

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\nPipeline Error: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()