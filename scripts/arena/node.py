import math

from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from .config import DEVICES


class ArenaController(Node):
    def __init__(self):
        super().__init__('arena_controller')
        
        self.devices = DEVICES
        
        # Initialize publishers for BOTH Simulation and Real Life simultaneously
        self.pubs = {'sim': {}, 'real': {}}
        for name, config in self.devices.items():
            self.pubs['sim'][name] = self.create_publisher(JointState, config['topic'], 10)
            self.pubs['real'][name] = self.create_publisher(JointState, config['topic'].replace('/sim/', '/'), 10)
            
        # --- TV STREAM PUBLISHERS ---
        self.tv_pub_sim = self.create_publisher(String, '/sim/tv/stream_command', 10)
        self.tv_pub_real = self.create_publisher(String, '/tv/stream_command', 10)
            
        self.get_logger().info("Arena Controller is ready in DUAL MODE (Sim & Real Life)!")

    def set_position(self, device_name, position_input, target_mode):
        """Builds and publishes the JointState message to the correct environment."""
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
                
            log_msg = f"---> [{target_mode.upper()}] Commanded {device_name.upper()} to {position_final * 100:.1f} cm ({position_final:.3f} m)."

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
                
            log_msg = f"---> [{target_mode.upper()}] Commanded {device_name.upper()} to {math.degrees(position_final):.1f}° ({position_final:.2f} rad)."

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
        
        # Publish to the specific target environment
        self.pubs[target_mode][device_name].publish(msg)
        self.get_logger().info(log_msg)

    def set_tv_stream(self, stream_topic, target_mode):
        """Publishes the string command to change the TV stream in the correct environment."""
        msg = String()
        msg.data = stream_topic
        
        if target_mode == 'sim':
            self.tv_pub_sim.publish(msg)
        else:
            self.tv_pub_real.publish(msg)
            
        self.get_logger().info(f"---> [{target_mode.upper()}] Commanded TV to stream: {stream_topic}")
