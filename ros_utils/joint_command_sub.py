import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import matplotlib.pyplot as plt
import numpy as np
import os

PANDA_JOINT_NAMES = [
    'panda_joint1', 'panda_joint2', 'panda_joint3',
    'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
]

class JointPlotterNode(Node):
    def __init__(self):
        super().__init__('joint_plotter_node')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_command',
            self.listener_callback,
            10
        )
        self.joint_data = {name: [] for name in PANDA_JOINT_NAMES}
        self.step_count = 0

    def listener_callback(self, msg: JointState):
        self.get_logger().info(f"[{self.step_count}] Received joint command with names: {msg.name}")
        name_to_pos = dict(zip(msg.name, msg.position))
        for joint in PANDA_JOINT_NAMES:
            if joint in name_to_pos:
                self.joint_data[joint].append(name_to_pos[joint])
        self.step_count += 1

    def plot_and_save(self):
        if self.step_count == 0:
            self.get_logger().warn("No joint data received. Plot will not be generated.")
            return

        fig, ax = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [4, 1]})
        time_axis = np.arange(self.step_count)

        # Plot joint values
        for joint in PANDA_JOINT_NAMES:
            data = self.joint_data[joint]
            if len(data) > 0:
                ax[0].plot(time_axis[:len(data)], data, label=joint)

        ax[0].set_title("Joint Position Commands Over Time")
        ax[0].set_xlabel("Step")
        ax[0].set_ylabel("Joint Position (rad)")
        ax[0].legend(loc='upper right')
        ax[0].grid(True)

        # Compute stats
        stats = []
        for joint in PANDA_JOINT_NAMES:
            data = np.array(self.joint_data[joint])
            if len(data) == 0:
                stats.append(["N/A", "N/A", "N/A"])
                continue
            mean = np.mean(data)
            std = np.std(data)
            min_val = np.min(data)
            max_val = np.max(data)
            stats.append([f"{mean:.3f}", f"{std:.3f}", f"[{min_val:.3f}, {max_val:.3f}]"])

        # Add table to bottom axis
        table = ax[1].table(
            cellText=stats,
            rowLabels=PANDA_JOINT_NAMES,
            colLabels=["Mean", "Std", "Range"],
            loc='center'
        )
        table.scale(1, 2)
        ax[1].axis('off')
        ax[1].set_title("Joint Statistics")

        plt.tight_layout()
        save_path = os.path.join(os.getcwd(), "joint_stats_plot.png")
        plt.savefig(save_path)
        self.get_logger().info(f"Plot saved to {save_path}")

def main(args=None):
    rclpy.init(args=args)
    node = JointPlotterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received. Saving plot and shutting down...")
    finally:
        node.plot_and_save()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

