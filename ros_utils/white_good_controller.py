import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time
import math


class WhiteGoodController(Node):
    def __init__(self):
        super().__init__('interactive_appliance_controller')
        
        # Configuration for all your appliances (Limits kept in radians as ROS standard)
        self.appliances = {
            'fridge': {
                'topic': '/fridge/joint_command', 
                'joint': 'fridge_joint', 
                'limit': math.radians(-170) # Updated to 170 degrees
            },
            'dishwasher': {
                'topic': '/dishwasher/joint_command', 
                'joint': 'dishwasher_joint', 
                'limit': 1.5
            },
            'washing_machine': {
                'topic': '/washing_machine/joint_command', 
                'joint': 'washing_machine_joint', 
                'limit': 1.5
            },
            'oven': {
                'topic': '/oven/joint_command', 
                'joint': 'oven_joint', 
                'limit': 1.5
            }
        }
        
        # Create publishers dynamically
        self.pubs = {}
        for name, config in self.appliances.items():
            self.pubs[name] = self.create_publisher(JointState, config['topic'], 10)
            
        self.get_logger().info("Interactive Appliance Controller is ready!")

    def set_position(self, appliance_name, position_degrees):
        """Builds and publishes the JointState message, handling hardware conversions and limits."""
        config = self.appliances[appliance_name]
        
        # Hardware Abstraction: The fridge opens natively in the negative direction.
        if appliance_name == 'fridge':
            position_degrees = -abs(position_degrees)
        
        # Convert the degree input to radians for ROS
        position_radians = math.radians(position_degrees)
        
        # Safety Check & Clamping: If commanded position exceeds limits, clamp it
        if abs(position_radians) > abs(config['limit']):
            limit_deg = abs(math.degrees(config['limit']))
            self.get_logger().warning(
                f"Command ({abs(position_degrees)}°) exceeds {appliance_name} limit ({limit_deg:.1f}°). Clamping to maximum limit."
            )
            # Clamp to the maximum limit while preserving the correct direction (sign)
            position_radians = math.copysign(abs(config['limit']), position_radians)
            position_degrees = math.degrees(position_radians) # Update degrees for the final log message

        # Build the message
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [config['joint']]
        msg.position = [float(position_radians)]  # ROS standard expects radians
        
        # Publish
        self.pubs[appliance_name].publish(msg)
        self.get_logger().info(f"---> Successfully commanded {appliance_name} to {abs(position_degrees):.1f}° ({position_radians:.2f} rad).\n")


def main(args=None):
    rclpy.init(args=args)
    controller = WhiteGoodController()

    # Give publishers a moment to establish connections
    time.sleep(0.5) 

    print("\n" + "="*50)
    print("        🏠 SMART HOME JOINT CONTROLLER 🏠        ")
    print("="*50)

    try:
        # Create a static list of appliances for the menu
        appliances_list = list(controller.appliances.keys())

        # The Interactive Loop
        while rclpy.ok():
            print("\n--- Select an Appliance ---")
            for i, name in enumerate(appliances_list, 1):
                # Format the name nicely (e.g., 'washing_machine' -> 'Washing Machine')
                print(f"{i}. {name.replace('_', ' ').title()}")
            print("0. Quit")
            
            # 1. Get the appliance selection via numbered menu
            choice_str = input("\nEnter choice (0-4): ").strip()
            
            if choice_str == '0' or choice_str.lower() in ['q', 'quit', 'exit']:
                print("Exiting controller. Goodbye!")
                break
                
            try:
                choice = int(choice_str)
                if choice < 1 or choice > len(appliances_list):
                    raise ValueError
                appliance_name = appliances_list[choice - 1]
            except ValueError:
                print("[!] Error: Invalid selection. Please enter a valid number from the menu.\n")
                continue
                
            # 2. Get the position in DEGREES
            limit_rad = controller.appliances[appliance_name]['limit']
            limit_deg = abs(math.degrees(limit_rad))
            
            pos_input = input(f"Enter opening amount in DEGREES for {appliance_name} (Max limit: {limit_deg:.1f}°): ").strip()
            
            try:
                position_degrees = float(pos_input)
            except ValueError:
                print("[!] Error: Position must be a valid number. Please try again.\n")
                continue
                
            # 3. Execute the command
            controller.set_position(appliance_name, position_degrees)
            
            # 4. Spin once to process the publisher queue before the next loop
            rclpy.spin_once(controller, timeout_sec=0.1)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Shutting down...")

    finally:
        controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()