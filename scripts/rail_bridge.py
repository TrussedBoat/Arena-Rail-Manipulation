#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import copy

class RailUnifiedBridge(Node):
    def __init__(self):
        super().__init__('rail_unified_bridge_node')
        
        # Cache for last known joint states (to fill partial commands)
        self.last_state1 = JointState()
        self.last_state2 = JointState()
        
        # Cache for latest commanded joint positions and velocities from /joint_command
        self.sys1_cmd_cache = {f"panda_joint{i}": 0.0 for i in range(1, 8)}
        self.sys1_cmd_cache["panda_finger_joint1"] = 0.0
        
        self.sys1_cmd_vel_cache = {f"panda_joint{i}": 0.0 for i in range(1, 8)}
        self.sys1_cmd_vel_cache["panda_finger_joint1"] = 0.0
        self.sys1_cmd_received = set()
        
        self.sys2_cmd_cache = {f"panda_joint{i}": 0.0 for i in range(1, 8)}
        self.sys2_cmd_cache["panda_finger_joint1"] = 0.0
        
        self.sys2_cmd_vel_cache = {f"panda_joint{i}": 0.0 for i in range(1, 8)}
        self.sys2_cmd_vel_cache["panda_finger_joint1"] = 0.0
        self.sys2_cmd_received = set()
        
        # ==========================================
        # SPLITTER: Isaac Sim (Bundled) -> Controllers (Separated)
        # ==========================================
        self.sub_state1 = self.create_subscription(
            JointState, '/sim/rail_franka1/joint_states', self.cb_split_state1, 10)
        self.sub_state2 = self.create_subscription(
            JointState, '/sim/rail_franka2/joint_states', self.cb_split_state2, 10)
            
        self.pub_rail1_state = self.create_publisher(JointState, '/sim/rail1/joint_states', 10)
        self.pub_franka1_state = self.create_publisher(JointState, '/joint_states', 10)
        self.pub_franka1_gripper_state = self.create_publisher(Float64, '/gripper_state', 10)
        
        self.pub_rail2_state = self.create_publisher(JointState, '/sim/rail2/joint_states', 10)
        self.pub_franka2_state = self.create_publisher(JointState, '/sim/franka2_rail/joint_states', 10)

        # ==========================================
        # RELAY: Controllers (Separated) -> Isaac Sim (Bundled)
        # ==========================================
        self.pub_sys1_cmd = self.create_publisher(JointState, '/sim/rail_franka1/joint_command', 10)
        self.pub_sys2_cmd = self.create_publisher(JointState, '/sim/rail_franka2/joint_command', 10)
        
        self.create_subscription(JointState, '/sim/rail1/joint_command', self.cb_sys1_relay, 10)
        self.create_subscription(JointState, '/joint_command', self.cb_sys1_relay, 10)
        self.create_subscription(JointState, '/joint_position_command', self.cb_sys1_relay, 10)
        self.create_subscription(Float64, '/gripper_cmd', self.cb_sys1_relay, 10)
        self.create_subscription(JointState, '/sim/rail2/joint_command', self.cb_sys2_relay, 10)
        self.create_subscription(JointState, '/sim/franka2_rail/joint_command', self.cb_sys2_relay, 10)

        self.get_logger().info("Unified Splitter/Relay Bridge Active with Position and Velocity Caching.")

    # ---------------------------------------------------------
    # MERGE LOGIC
    # ---------------------------------------------------------
    def cb_sys1_relay(self, msg):
        # Update the command cache with incoming values (position and velocity)
        if hasattr(msg, 'name'):
            # msg is a JointState message
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    self.sys1_cmd_cache[name] = msg.position[i]
                    self.sys1_cmd_received.add(name)
                if i < len(msg.velocity):
                    self.sys1_cmd_vel_cache[name] = msg.velocity[i]
        else:
            # msg is a Float64 message from /gripper_cmd (position only usually)
            self.sys1_cmd_cache["panda_finger_joint1"] = msg.data
            self.sys1_cmd_received.add("panda_finger_joint1")

        # Construct and publish a unified command message containing all cached joints
        out_msg = JointState()
        if hasattr(msg, 'header'):
            out_msg.header = msg.header
        else:
            out_msg.header.stamp = self.get_clock().now().to_msg()
            
        for name in self.sys1_cmd_cache.keys():
            out_msg.name.append(name)
            out_msg.position.append(self.sys1_cmd_cache[name])
            out_msg.velocity.append(self.sys1_cmd_vel_cache[name])
            
        self.pub_sys1_cmd.publish(out_msg)

    def cb_sys2_relay(self, msg):
        # Update the command cache with incoming values (position and velocity)
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.sys2_cmd_cache[name] = msg.position[i]
                self.sys2_cmd_received.add(name)
            if i < len(msg.velocity):
                self.sys2_cmd_vel_cache[name] = msg.velocity[i]

        # Construct and publish a unified command message containing all cached joints
        out_msg = JointState()
        out_msg.header = msg.header
        for name in self.sys2_cmd_cache.keys():
            out_msg.name.append(name)
            out_msg.position.append(self.sys2_cmd_cache[name])
            out_msg.velocity.append(self.sys2_cmd_vel_cache[name])
            
        self.pub_sys2_cmd.publish(out_msg)

    # ---------------------------------------------------------
    # SPLITTER LOGIC
    # ---------------------------------------------------------
    def process_and_split_state(self, msg, rail_pub, franka_pub, cache_attr):
        setattr(self, cache_attr, msg) # Update the cache
        
        # Update command cache with exact, actual joint positions from the simulation
        cmd_cache = self.sys1_cmd_cache if cache_attr == 'last_state1' else self.sys2_cmd_cache
        cmd_vel_cache = self.sys1_cmd_vel_cache if cache_attr == 'last_state1' else self.sys2_cmd_vel_cache
        cmd_received = self.sys1_cmd_received if cache_attr == 'last_state1' else self.sys2_cmd_received
        
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                if name not in cmd_received:
                    cmd_cache[name] = msg.position[i]
                    # Also default velocity to the actual simulation velocity (or 0) if no command received
                    if msg.velocity and i < len(msg.velocity):
                        cmd_vel_cache[name] = msg.velocity[i]
                
                # If this is System 1's gripper finger joint, publish it to /gripper_state
                if name == 'panda_finger_joint1' and cache_attr == 'last_state1':
                    gripper_msg = Float64()
                    gripper_msg.data = msg.position[i]
                    self.pub_franka1_gripper_state.publish(gripper_msg)
        
        rail_msg = JointState()
        rail_msg.header = msg.header
        franka_msg = JointState()
        franka_msg.header = msg.header
        
        for i, name in enumerate(msg.name):
            if 'rail' in name.lower():
                rail_msg.name.append(name)
                if msg.position: rail_msg.position.append(msg.position[i])
                if msg.velocity: rail_msg.velocity.append(msg.velocity[i])
                if msg.effort: rail_msg.effort.append(msg.effort[i])
            elif 'panda' in name.lower():
                franka_msg.name.append(name)
                if msg.position: franka_msg.position.append(msg.position[i])
                if msg.velocity: franka_msg.velocity.append(msg.velocity[i])
                if msg.effort: franka_msg.effort.append(msg.effort[i])

        if rail_msg.name: rail_pub.publish(rail_msg)
        if franka_msg.name: franka_pub.publish(franka_msg)

    def cb_split_state1(self, msg):
        self.process_and_split_state(msg, self.pub_rail1_state, self.pub_franka1_state, 'last_state1')

    def cb_split_state2(self, msg):
        self.process_and_split_state(msg, self.pub_rail2_state, self.pub_franka2_state, 'last_state2')

def main(args=None):
    rclpy.init(args=args)
    node = RailUnifiedBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
