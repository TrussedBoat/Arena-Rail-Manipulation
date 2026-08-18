#!/usr/bin/env python3
from pathlib import Path
import sys
import json

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool, Empty, Float64, String
import copy

# This script is launched directly, so expose the orchestrator source directory
# before importing the shared runtime configuration.
ORCHESTRATOR_SRC = Path(__file__).resolve().parents[1] / 'agent_orchestrator' / 'src'
if str(ORCHESTRATOR_SRC) not in sys.path:
    sys.path.insert(0, str(ORCHESTRATOR_SRC))

import ros_logger
ros_logger.setup_ros_logging()

from config import load_runtime_config

DIRECT_TARGET_STABLE_SAMPLES = 5

class RailUnifiedBridge(Node):
    def __init__(self):
        super().__init__('rail_unified_bridge_node')
        self.direct_target_position_tolerance = (
            load_runtime_config().search.wrist_joint_tolerance
        )
        
        # Cache for last known joint states (to fill partial commands)
        self.last_state1 = JointState()
        self.last_state2 = JointState()
        
        # Cache for latest planned Cartesian joint positions and velocities.
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
        self.cartesian_control_active = False
        self.direct_joint_control_active = False
        self.direct_target_positions = {}
        self.direct_target_stable_samples = 0
        
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
        direct_control_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.direct_control_state_pub = self.create_publisher(
            Bool, '/panda/direct_joint_control_active', direct_control_qos
        )
        
        self.create_subscription(JointState, '/sim/rail1/joint_command', self.cb_sys1_direct_relay, 10)
        self.create_subscription(JointState, '/cartesian/joint_command', self.cb_sys1_controller_relay, 10)
        self.create_subscription(JointState, '/direct_joint_command', self.cb_sys1_direct_relay, 10)
        # RRT publishes this only for cancellation/safety holds.  It must
        # reach the simulator as a direct command, but stays separate from
        # the VLM/orchestrator direct-joint channel.
        self.create_subscription(JointState, '/rrt/hold_command', self.cb_sys1_direct_relay, 10)
        self.create_subscription(Float64, '/gripper/command', self.cb_sys1_direct_relay, 10)
        self.create_subscription(Pose, '/rrt/pose_command', self.cb_cartesian_pose, 10)
        self.create_subscription(Empty, '/rrt/reverse_last_trajectory', self.cb_reverse_trajectory, 10)
        self.create_subscription(String, '/rrt/status', self.cb_rrt_status, 10)
        self.create_subscription(JointState, '/sim/rail2/joint_command', self.cb_sys2_relay, 10)
        self.create_subscription(JointState, '/sim/franka2_rail/joint_command', self.cb_sys2_relay, 10)

        self.get_logger().info(
            "Unified Splitter/Relay Bridge Active with Position and Velocity "
            f"Caching (direct Panda tolerance="
            f"{self.direct_target_position_tolerance:.4f}rad)."
        )
        self._publish_direct_control_state()

    # ---------------------------------------------------------
    # MERGE LOGIC
    # ---------------------------------------------------------
    def cb_cartesian_pose(self, _msg):
        if self.direct_joint_control_active:
            self.get_logger().warning(
                "Ignoring Cartesian pose: direct Panda joint control has priority."
            )
            return
        if not self.cartesian_control_active:
            self.cartesian_control_active = True
            self.get_logger().info(
                "Cartesian arm ownership enabled; direct Panda/gripper commands are now blocked."
            )

    def cb_rrt_status(self, msg):
        """Release bridge ownership only when the planner ends a trajectory."""
        try:
            state = str(json.loads(msg.data).get('state', ''))
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if state not in {'success', 'cancelled'} or not self.cartesian_control_active:
            return
        self.cartesian_control_active = False
        self.get_logger().info(
            f"RRT trajectory {state}; Cartesian ownership released for direct gripper commands."
        )

    def cb_reverse_trajectory(self, _msg):
        if self.direct_joint_control_active:
            self.get_logger().warning(
                "Ignoring reverse RRT trajectory: direct Panda joint control has priority."
            )
            return
        self.cartesian_control_active = True
        self.get_logger().info("Cartesian ownership enabled for reverse RRT trajectory replay.")

    @staticmethod
    def _is_panda_joint(name):
        return 'panda' in name.lower()

    def _publish_direct_control_state(self):
        message = Bool()
        message.data = self.direct_joint_control_active
        self.direct_control_state_pub.publish(message)

    def cb_sys1_controller_relay(self, msg):
        if self.direct_joint_control_active:
            self.get_logger().warning(
                "Ignoring Cartesian joint command: direct Panda joint control has priority."
            )
            return
        if not self.cartesian_control_active:
            # Do not replay a stale Cartesian target after direct-joint ownership
            # has completed. A new /rrt/pose_command explicitly arms Cartesian forwarding.
            return
        self._relay_sys1_command(msg, controller_owned=True)

    def cb_sys1_direct_relay(self, msg):
        if hasattr(msg, 'name'):
            panda_names = [name for name in msg.name if self._is_panda_joint(name)]
            if panda_names:
                if not self.direct_joint_control_active:
                    self.get_logger().warning(
                        "Direct Panda joint control enabled; Cartesian commands are now blocked."
                    )
                self.direct_joint_control_active = True
                self._publish_direct_control_state()
                self.cartesian_control_active = False
                self.direct_target_positions = {
                    name: float(msg.position[index])
                    for index, name in enumerate(msg.name)
                    if self._is_panda_joint(name)
                    and index < len(msg.position)
                    and np.isfinite(msg.position[index])
                }
                self.direct_target_stable_samples = 0
        self._relay_sys1_command(msg, controller_owned=False)

    def _check_direct_target_convergence(self, msg):
        """Release direct ownership after a stable, in-tolerance joint target."""
        if not self.direct_joint_control_active or not self.direct_target_positions:
            return

        actual_positions = {
            name: float(msg.position[index])
            for index, name in enumerate(msg.name)
            if name in self.direct_target_positions and index < len(msg.position)
        }
        if len(actual_positions) != len(self.direct_target_positions):
            self.direct_target_stable_samples = 0
            return

        converged = all(
            abs(actual_positions[name] - target)
            <= self.direct_target_position_tolerance
            for name, target in self.direct_target_positions.items()
        )
        if not converged:
            self.direct_target_stable_samples = 0
            return

        self.direct_target_stable_samples += 1
        if self.direct_target_stable_samples < DIRECT_TARGET_STABLE_SAMPLES:
            return

        self.direct_joint_control_active = False
        self._publish_direct_control_state()
        self.direct_target_positions.clear()
        self.direct_target_stable_samples = 0
        self.get_logger().info(
            "Direct Panda target reached; Cartesian command forwarding re-enabled."
        )

    def _relay_sys1_command(self, msg, controller_owned):
        if self.cartesian_control_active and not controller_owned:
            if not hasattr(msg, 'name'):
                self.get_logger().warning(
                    "Ignoring direct gripper command: Cartesian controller owns Panda joints."
                )
                return

            allowed_indices = [
                index for index, name in enumerate(msg.name)
                if not self._is_panda_joint(name)
            ]
            blocked = [
                name for name in msg.name if self._is_panda_joint(name)
            ]
            if blocked:
                self.get_logger().warning(
                    "Ignoring direct Cartesian-owned joints: " + ", ".join(blocked)
                )
            if not allowed_indices:
                return

            filtered = JointState()
            filtered.header = msg.header
            for index in allowed_indices:
                filtered.name.append(msg.name[index])
                if index < len(msg.position):
                    filtered.position.append(msg.position[index])
                if index < len(msg.velocity):
                    filtered.velocity.append(msg.velocity[index])
                if index < len(msg.effort):
                    filtered.effort.append(msg.effort[index])
            msg = filtered

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
            # msg is a Float64 message from /gripper/command (position only usually)
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
            out_msg.velocity.append(self.sys1_cmd_vel_cache.get(name, 0.0))
            
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
            out_msg.velocity.append(self.sys2_cmd_vel_cache.get(name, 0.0))
            
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
        self._check_direct_target_convergence(msg)

    def cb_split_state2(self, msg):
        self.process_and_split_state(msg, self.pub_rail2_state, self.pub_franka2_state, 'last_state2')

def main(args=None):
    rclpy.init(args=args)
    node = RailUnifiedBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
