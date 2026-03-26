import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time
import math


class ArenaController(Node):
    def __init__(self):
        super().__init__('arena_controller')
        
        self.devices = {
            # --- WHITE GOODS ---
            'fridge': {
                'topic': '/sim/fridge/joint_command', 
                'joint': 'fridge_joint', 
                'min': math.radians(-170),
                'max': 0.0,
                'unit': 'degrees',
                'category': 'White Goods'
            },
            'dishwasher': {
                'topic': '/sim/dishwasher/joint_command', 
                'joint': 'dishwasher_joint', 
                'min': 0.0,
                'max': 1.5,
                'unit': 'degrees',
                'category': 'White Goods'
            },
            'washing_machine': {
                'topic': '/sim/washing_machine/joint_command', 
                'joint': 'washing_machine_joint', 
                'min': 0.0,
                'max': 1.5,
                'unit': 'degrees',
                'category': 'White Goods'
            },
            'oven': {
                'topic': '/sim/oven/joint_command', 
                'joint': 'oven_joint', 
                'min': 0.0,
                'max': 1.5,
                'unit': 'degrees',
                'category': 'White Goods'
            },
            
            # --- RAILS ---
            'franka_rail1': {
                'topic': '/sim/rail_franka1/joint_command', 
                'joint': 'rail_j1', 
                'min': -1.5,
                'max': 2.5,
                'unit': 'meters',
                'category': 'Rails'
            },
            'franka_rail2': {
                'topic': '/sim/rail_franka2/joint_command', 
                'joint': 'rail_j2', 
                'min': -2.9,
                'max': 0.9,
                'unit': 'meters',
                'category': 'Rails'
            }
        }
        
        # --- TABLES ---
        for i in range(8):
            self.devices[f't{i}'] = {
                'topic': f'/sim/t{i}/joint_command',
                'joint': ['j1', 'j2'], 
                'min': 0.0,
                'max': 0.65,
                'unit': 'meters',
                'category': 'Tables'
            }
        
        self.pubs = {}
        for name, config in self.devices.items():
            self.pubs[name] = self.create_publisher(JointState, config['topic'], 10)
            
        self.get_logger().info("Arena Controller is ready!")

    def set_position(self, device_name, position_input):
        """Builds and publishes the JointState message."""
        config = self.devices[device_name]
        is_linear = (config['unit'] == 'meters')
        
        if is_linear:
            # Linear (Input in CM, publish in Meters)
            position_final = position_input / 100.0
            
            if position_final < config['min']:
                self.get_logger().warning(f"Command ({position_input}cm) below minimum. Clamping to {config['min']*100:.0f}cm.")
                position_final = config['min']
            elif position_final > config['max']:
                self.get_logger().warning(f"Command ({position_input}cm) exceeds maximum. Clamping to {config['max']*100:.0f}cm.")
                position_final = config['max']
                
            log_msg = f"---> Commanded {device_name.upper()} to {position_final * 100:.1f} cm ({position_final:.3f} m)."

        else:
            # Rotational
            position_degrees = position_input
            if device_name == 'fridge':
                position_degrees = -abs(position_degrees)
            
            position_final = math.radians(position_degrees)
            
            if position_final < config['min']:
                min_deg = math.degrees(config['min'])
                self.get_logger().warning(f"Command exceeds limits. Clamping to {min_deg:.1f}°.")
                position_final = config['min']
            elif position_final > config['max']:
                max_deg = math.degrees(config['max'])
                self.get_logger().warning(f"Command exceeds limits. Clamping to {max_deg:.1f}°.")
                position_final = config['max']
                
            log_msg = f"---> Commanded {device_name.upper()} to {math.degrees(position_final):.1f}° ({position_final:.2f} rad)."

        # Build & Publish
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        joints = config['joint']
        if isinstance(joints, list):
            msg.name = joints
            msg.position = [float(position_final)] * len(joints)
        else:
            msg.name = [joints]
            msg.position = [float(position_final)]
        
        self.pubs[device_name].publish(msg)
        self.get_logger().info(log_msg)


def print_ui_header():
    print("\n" + "═"*70)
    print(" 🤖 ARENA CONTROLLER: SEQUENCE BUILDER 🤖 ".center(68, "═"))
    print("═"*70)
    print("\n ⚠️  OPERATIONAL GUIDELINES:")
    print(" ┌──────────────────────────────────────────────────────────────────┐")
    print(" │ 🧊 WHITE GOODS : Elevate adjacent tables before opening doors!   │")
    print(" │ 🛤️  ROBOT RAILS : Absolute coordinates. (+) moves toward wall.   │")
    print(" │ 🪑 TABLES (0-7): Wall Side: T0/T1 | Sitting Area Side: T6/T7     │")
    print(" │                  Odd: Door side   | Even: Window side            │")
    print(" └──────────────────────────────────────────────────────────────────┘\n")


def print_queue(queue):
    print("\n" + "📋 CURRENT SEQUENCE QUEUE ".ljust(70, "-"))
    if not queue:
        print("   (Queue is empty)")
    else:
        for i, cmd in enumerate(queue, 1):
            if cmd['type'] == 'device':
                print(f"   {i}. Set [ {cmd['device'].upper()} ] -> {cmd['value']} {cmd['unit']}")
            elif cmd['type'] == 'wait':
                print(f"   {i}. ⏱️  WAIT for {cmd['value']} seconds")
    print("-" * 70 + "\n")


def main(args=None):
    rclpy.init(args=args)
    controller = ArenaController()
    time.sleep(0.5) 
    print_ui_header()

    command_queue = []
    numeric_devices = [name for name, config in controller.devices.items() if config['category'] != 'Tables']

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
            print("  [t0-t7] Type table name directly (e.g., 't0')")
            
            print("\n\n" + " SEQUENCE CONTROLS ".center(40, "=") + "\n")
            print(f"  [W] Add Wait/Delay  |  Queue size: {len(command_queue)}")
            print("  [E] Execute Queue   |  [V] View Queue")
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
                        controller.set_position(step['device'], step['value'])
                        rclpy.spin_once(controller, timeout_sec=0.05)
                    elif step['type'] == 'wait':
                        print(f"---> ⏱️  Waiting for {step['value']} seconds...")
                        time.sleep(step['value'])
                
                print("═"*70)
                print("✅ Sequence Complete!")
                command_queue.clear() # Clear after successful execution
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
                
                # Append to queue instead of executing
                command_queue.append({
                    'type': 'device', 
                    'device': device_name, 
                    'value': pos_input, 
                    'unit': unit_disp
                })
                print(f"\n[+] Added {device_name.upper()} -> {pos_input}{unit_disp} to queue.")
                
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