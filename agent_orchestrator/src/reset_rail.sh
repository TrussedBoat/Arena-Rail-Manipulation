#!/bin/bash
# initialize_robot.sh: Sets the robot to the known home state
# Usage: ros2 topic pub -1 /topic msg "{data}"

source /opt/ros/humble/setup.bash 

# 1. Move Rails to -1.1
ros2 topic pub -1 /sim/rail_franka1/joint_command sensor_msgs/msg/JointState "{
  name: ['rail_j1'],
  position: [-1.1]
}"

# 2. Set Arm Poses [0, -45, 0, -135, 0, 90, 45] (Converted to radians)
ros2 topic pub -1 /sim/rail_franka1/joint_command sensor_msgs/msg/JointState "{
  name: ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'],
  position: [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854]
}"

# 3. Open Gripper
ros2 topic pub -1 /gripper_command std_msgs/msg/Float64 "{data: 100.0}"

echo "Robot initialization complete."