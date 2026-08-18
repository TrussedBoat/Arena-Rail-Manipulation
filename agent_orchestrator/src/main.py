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
                        "Plan adaptively from semantic map evidence and current camera evidence; do not follow a fixed object-by-object search script.\n\n"
                        "NON-NEGOTIABLE SAFETY:\n"
                        "- Call start_joint_controller first. Call exactly one high-level tool per step.\n"
                        "- Immediately before EVERY execute_pick_script or execute_place_script call, call get_camera_frame. It waits two seconds, takes two new ROS wrist-camera frames, and supplies the later one. In the next <think>, state what is visibly present, including colors, and explicitly confirm the required pickup or destination object. Do not execute if it is absent or unclear; do not infer visual qualifiers from the semantic map.\n"
                        "- Manipulate the pickup object last among the verification moves so the robot is positioned at it. A targeted_search result with state=couldnt_find is recoverable planning evidence: decide whether broader mapping is justified or finish with an honest failure. Do not execute a dependent manipulation while a required object remains unconfirmed.\n\n"
                        "SEMANTIC-FIRST PLANNING:\n"
                        "1. In the first planning step after initialization, identify all important object classes in the user task and call get_semantic_objects with their canonical base classes. For example, query bowl for 'purple bowl'. Include objects needed to resolve spatial language such as closest, left/right, next to, or between.\n"
                        "2. The returned JSON is map evidence: use confirmed XYZ positions to select objects, compare spatial relations, and choose an efficient approach order. A candidate, class_ambiguous, stale, or absent entry is insufficient. The map does not prove color, shape details, or physical presence.\n"
                        "3. If all information needed for the task is confirmed, select exact object_id values from the map and use move_rail_to_object(object_id). Use those same IDs for pick and destination placement. Do not move merely to discover whether an object exists.\n"
                        "4. Choose the search scale from the missing context. Use targeted_search for one missing/non-confirmed class when locating that class alone lets the task continue. Use general_mapping when the request requires scene-wide or relational context that is unavailable, or when two or more relevant classes are missing/non-confirmed.\n"
                        "5. After targeted_search or general_mapping, call get_semantic_objects again for the relevant classes and re-plan from the refreshed JSON. Never assume a search result resolves a relation without examining the updated positions.\n\n"
                        "EXECUTION:\n"
                        "Use the map to decide what to inspect and where to go, then use camera frames for physical verification before manipulation. Complete the task with finish_task only after the requested action succeeds."
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
