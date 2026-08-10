#!/usr/bin/env python3
import time
import re
import rclpy
import subprocess
import argparse
import ros_logger
ros_logger.setup_ros_logging()

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from config import load_runtime_config, set_runtime_config, validate_runtime_config

# max 8 gb kullanan bir vlm modeli bul ve entegre et
# basit ama net,  bir system prompt tasarla, sadece güncel görüntüyü alıp arama başlatacak şekilde
# uzun bir system promptu ve fazla tool tanımı vlm in performansını düşürür
# arama yaparken bir yandan robotu hareket ettirmeli bir yandan da nesneyi takip etmeye çalışmalı
# nesne tespit edilirse hareket ettirmeyi bırakıo nesnenin yerini referans frame e göre hesaplayıp json a kaydetmeli
# o referans frame in tanımı kritik ve robust olmalı



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-think", action="store_true", help="Do not filter out <think> tags from the terminal output")
    args, _ = parser.parse_known_args()

    tools = None
    runtime_started = False
    ros_initialized = False

    try:
        import tools as runtime_tools

        tools = runtime_tools
        config = load_runtime_config()
        validate_runtime_config(config)
        set_runtime_config(config)
        print(
            "[SYSTEM] Runtime configuration validated: "
            f"{config.vlm.model_alias}, context={config.vlm.context_size}, "
            f"VLM VRAM budget<={config.vlm.vram_budget_gb:g}GB, "
            f"YOLO={config.yolo.checkpoint_path.name}."
        )

        rclpy.init()
        ros_initialized = True
        runtime_started = True
        tools.start_vlm_server(config)

        # Importing agent constructs ChatOpenAI, so do it only after configuration
        # and server readiness have passed their fail-fast gates.
        from agent import agent
        
        print("Starting interactive dynamic agentic-tool pipeline...")
        print("Type 'quit' or 'exit' at any time to stop the script.\n")
        
        while True:
            user_task = input("\n[USER] Enter your next task: ").strip()
            
            if user_task.lower() in ['quit', 'exit', 'q']:
                print("Shutting down pipeline...")
                break

            if user_task == "":
                user_task = "pick up the apple and place it into the purple bowl"
                
            if not user_task:
                continue
                        
            input_state = {
                "messages": [
                    SystemMessage(content=(
                        "You are a robotic task orchestrator operating via JSON tools.\n"
                        "You have the freedom to plan the necessary steps to complete the task.\n\n"
                        "CRITICAL PIPELINE RULES:\n"
                        "1. INITIALIZATION: You MUST call 'start_joint_controller' as your very first step to activate the robot hardware.\n"
                        "2. REASONING & PLANNING: In your very first action's <think> block, you MUST explicitly list all items/objects concerned in the task. Before any pick or place script executes, every concerned object MUST have a known location and its physical presence MUST be visually confirmed via 'get_camera_frame'.\n"
                        "CAMERA EVIDENCE RULE: After every 'get_camera_frame' call and before deciding that an object is present or taking any action based on that frame, your next <think> block MUST ALWYAS describe what is visibly present: objects and colours. Also state explicitly whether the requested target is visible. If the image is unclear, missing, or does not visibly contain the target, state that it is not confirmed and do not infer its presence from the task or prior information.\n"
                        "3. PICK/PLACE ORDERING: The object to be manipulated (e.g., picked up) MUST be the LAST object confirmed so that the robot is physically positioned in front of it when the manipulation script is called.\n"
                        "4. COMPLETION: When the task is complete, call 'finish_task' with a completion summary.\n\n"
                        "OBJECT SEARCH & VERIFICATION WORKFLOW:\n"
                        "To locate and verify any object, you MUST strictly follow this loop:\n"
                        "  A. Call 'move_rail_to_object' to go to the object's known coordinates.\n"
                        "  B. If (A) succeeds, call 'get_camera_frame' to visually confirm the object is actually in the frame. If it is in the frame, the object is confirmed! If it is not in the frame, proceed to (C).\n"
                        "  C. If (A) returns an error or reports that the object is not in the semantic coordinates JSON OR if (B) fails (object not in frame), you MUST call 'targeted_search' for that object.\n"
                        "  D. If 'targeted_search' finds it, the object is confirmed! If 'targeted_search' fails to find it, the object is missing. You must then abort the task and return home.\n\n"
                        "EXECUTION RULES:\n"
                        "- Call exactly one tool per step.\n"
                        "- Stop immediately if any critical step fails without a fallback."
                    )),
                    HumanMessage(content=f"task: {user_task}")
                ],
                "iterations": 0,
                "terminated": False,
                "outcome": "in_progress",
                "tools_called": [],
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
                    if not args.show_think:
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
        return 1
    finally:
        if runtime_started:
            print("\n[CLEANUP] Initiating graceful shutdown sequence...")

            print(" -> Terminating 'global_joint_controller' tmux session...")
            subprocess.run(['tmux', 'kill-session', '-t', 'global_joint_controller'], capture_output=True)

            print(" -> Terminating 'rail_demo_pick' tmux session (failsafe)...")
            subprocess.run(['tmux', 'kill-session', '-t', 'rail_demo_pick'], capture_output=True)

            print(" -> Terminating 'rail_demo_place' tmux session (failsafe)...")
            subprocess.run(['tmux', 'kill-session', '-t', 'rail_demo_place'], capture_output=True)

            print(" -> Terminating 'vlm_server' tmux session (failsafe)...")
            if tools is not None:
                tools.stop_vlm_server()

            print(" -> Shutting down ROS 2 nodes...")
            if ros_initialized and rclpy.ok():
                rclpy.shutdown()
            print("[CLEANUP] Shutdown complete. Goodbye!")

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
