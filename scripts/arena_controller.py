import rclpy
import time
import math

from arena.config import CAMERA_TOPICS
from arena.node import ArenaController
from arena.ui import get_target_mode, print_ui_header, print_queue


def main(args=None):            
    rclpy.init(args=args)
    controller = ArenaController()
    time.sleep(0.5) 
    print_ui_header()

    command_queue = []
    numeric_devices = [name for name, config in controller.devices.items() if config['category'] != 'Tables']
    
    # --- BASE CAMERA TOPICS LIST ---
    camera_topics = CAMERA_TOPICS

    try:
        while rclpy.ok():
            print(" WHITE GOODS ".center(40, "-"))
            for i, name in enumerate(numeric_devices):
                if controller.devices[name]['category'] == 'White Goods':
                    print(f"  [{i+1}] {name.replace('_', ' ').title()}")
            
            print("\n" + " RAILS ".center(40, "-"))
            for i, name in enumerate(numeric_devices):
                if controller.devices[name]['category'] == 'Rails':
                    print(f"  [{i+1}] {name.replace('_', ' ').title()}")
            
            print("\n" + " TABLES ".center(40, "-"))
            print("  [t0-t7] Type table name directly (e.g., 't0') \n\n")
            
            print(" SEQUENCE CONTROLS ".center(40, "="))
            print(f"\n  [W] Add Wait/Delay  |  [T] Set TV Stream")
            print(f"  [E] Execute Queue   |  [V] View Queue   |  Queue size: {len(command_queue)}")
            print("  [C] Clear Queue     |  [0] Quit\n")
            print("═"*40)
            
            choice_str = input("Select an option: ").strip().lower()
            
            # --- QUEUE MANAGEMENT COMMANDS ---
            if choice_str in ['0', 'q', 'quit']:
                print("\nExiting Arena Controller...")
                break
                
            elif choice_str == 'v':
                print_queue(command_queue)
                input("Press Enter to continue...")
                continue
                
            elif choice_str == 'c':
                command_queue.clear()
                print("\n[!] Queue cleared.")
                time.sleep(1)
                continue
                
            elif choice_str == 'e':
                if not command_queue:
                    print("\n[!] Queue is empty. Add commands first.")
                    time.sleep(1)
                    continue
                
                print("\n🚀 EXECUTING SEQUENCE...")
                print("═"*70)
                for step in command_queue:
                    if step['type'] == 'device':
                        controller.set_position(step['device'], step['value'], step['target'])
                        rclpy.spin_once(controller, timeout_sec=0.05)
                    elif step['type'] == 'wait':
                        print(f"---> ⏱️  Waiting for {step['value']} seconds...")
                        time.sleep(step['value'])
                    elif step['type'] == 'tv_stream':
                        controller.set_tv_stream(step['value'], step['target'])
                        rclpy.spin_once(controller, timeout_sec=0.05)
                
                print("═"*70)
                print("✅ Sequence Complete!")
                command_queue.clear() 
                input("\nPress Enter to return to menu...")
                continue
                
            elif choice_str == 'w':
                try:
                    delay = float(input("\nEnter wait time in SECONDS: ").strip())
                    if delay < 0: raise ValueError
                    command_queue.append({'type': 'wait', 'value': delay})
                    print(f"\n[+] Added {delay}s wait to queue.")
                except ValueError:
                    print("\n[!] Error: Please enter a valid positive number.")
                time.sleep(1)
                continue
                
            elif choice_str == 't':
                print("\n" + " AVAILABLE CAMERAS ".center(60, "-"))
                for idx, topic in enumerate(camera_topics, 1):
                    print(f"  [{idx}] {topic}")
                print("  [0] Cancel")
                print("-" * 60)
                
                try:
                    cam_choice = int(input("\nSelect camera number: ").strip())
                    if cam_choice == 0:
                        print("\n[!] Stream selection cancelled.")
                    elif 1 <= cam_choice <= len(camera_topics):
                        raw_topic = camera_topics[cam_choice - 1]
                        
                        # Ask for target environment
                        target_mode = get_target_mode()
                        
                        # Strip '/sim' if targeting real life
                        stream_topic = raw_topic
                        if target_mode == 'real':
                            stream_topic = stream_topic.replace('/sim/', '/')
                            
                        # Overwrite if same target exists
                        found = False
                        for i, cmd in enumerate(command_queue):
                            if cmd['type'] == 'tv_stream' and cmd['target'] == target_mode:
                                command_queue[i] = {'type': 'tv_stream', 'value': stream_topic, 'target': target_mode}
                                found = True
                                print(f"\n[!] Updated existing TV Stream for {target_mode.upper()}.")
                                break
                        
                        if not found:
                            command_queue.append({'type': 'tv_stream', 'value': stream_topic, 'target': target_mode})
                            print(f"\n[+] Added TV Stream -> {stream_topic} (Target: {target_mode.upper()}) to queue.")
                    else:
                        print("\n[!] Error: Invalid selection. Please choose a number from the list.")
                except ValueError:
                    print("\n[!] Error: Please enter a valid number.")
                
                time.sleep(1)
                continue

            # --- DEVICE SELECTION COMMANDS ---
            device_name = None
            if choice_str in [f't{i}' for i in range(8)]:
                device_name = choice_str
            else:
                try:
                    choice = int(choice_str)
                    if 1 <= choice <= len(numeric_devices):
                        device_name = numeric_devices[choice - 1]
                    else:
                        raise ValueError
                except ValueError:
                    print("\n[!] Error: Invalid selection. Try again.")
                    time.sleep(1)
                    continue
                
            # --- GET POSITION FOR DEVICE ---
            config = controller.devices[device_name]
            print(f"\n--- Configuring {device_name.replace('_', ' ').title()} ---")
            
            try:
                if config['unit'] == 'meters':
                    min_cm = config['min'] * 100
                    max_cm = config['max'] * 100
                    pos_input = float(input(f"Enter position in CM ({min_cm:.0f} to {max_cm:.0f} cm): ").strip())
                    unit_disp = "cm"
                else:
                    min_deg = math.degrees(config['min'])
                    max_deg = math.degrees(config['max'])
                    display_limit = abs(min_deg) if device_name == 'fridge' else max_deg
                    pos_input = float(input(f"Enter opening in DEGREES (Max: {display_limit:.1f}°): ").strip())
                    unit_disp = "°"
                
                # Ask for target environment
                target_mode = get_target_mode()
                
                # Clamping logic
                if config['unit'] == 'meters':
                    min_cm = config['min'] * 100
                    max_cm = config['max'] * 100
                    if pos_input < min_cm:
                        print(f"\n[!] Command ({pos_input}cm) below minimum. Clamping to {min_cm:.0f}cm.")
                        pos_input = float(min_cm)
                    elif pos_input > max_cm:
                        print(f"\n[!] Command ({pos_input}cm) exceeds maximum. Clamping to {max_cm:.0f}cm.")
                        pos_input = float(max_cm)
                else:
                    # Rotational
                    min_deg = math.degrees(config['min'])
                    max_deg = math.degrees(config['max'])
                    if device_name == 'fridge':
                        max_opening = abs(min_deg)
                        if pos_input < 0:
                            print(f"\n[!] Command ({pos_input}°) below 0. Clamping to 0°.")
                            pos_input = 0.0
                        elif pos_input > max_opening:
                            print(f"\n[!] Command ({pos_input}°) exceeds maximum. Clamping to {max_opening:.1f}°.")
                            pos_input = float(max_opening)
                    else:
                        if pos_input < min_deg:
                            print(f"\n[!] Command ({pos_input}°) below minimum. Clamping to {min_deg:.1f}°.")
                            pos_input = float(min_deg)
                        elif pos_input > max_deg:
                            print(f"\n[!] Command ({pos_input}°) exceeds maximum. Clamping to {max_deg:.1f}°.")
                            pos_input = float(max_deg)

                # Overwrite logic
                found = False
                for i, cmd in enumerate(command_queue):
                    if cmd['type'] == 'device' and cmd['device'] == device_name and cmd['target'] == target_mode:
                        command_queue[i] = {
                            'type': 'device', 
                            'device': device_name, 
                            'value': pos_input, 
                            'unit': unit_disp,
                            'target': target_mode
                        }
                        found = True
                        print(f"\n[!] Updated existing {device_name.upper()} command for {target_mode.upper()}.")
                        break

                if not found:
                    command_queue.append({
                        'type': 'device', 
                        'device': device_name, 
                        'value': pos_input, 
                        'unit': unit_disp,
                        'target': target_mode
                    })
                    print(f"\n[+] Added {device_name.upper()} -> {pos_input}{unit_disp} (Target: {target_mode.upper()}) to queue.")
                
            except ValueError:
                print("\n[!] Error: Position must be a valid number.")
            
            time.sleep(1)
            print("\n" + "="*70 + "\n")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Shutting down...")
    finally:
        controller.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()