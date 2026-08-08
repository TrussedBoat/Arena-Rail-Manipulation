import logging
import rclpy.node
import os

# Create logger
ros_logger = logging.getLogger('ros_nodes')
ros_logger.setLevel(logging.DEBUG)
ros_logger.propagate = False

# Make sure we don't add handlers multiple times if imported multiple times
if not ros_logger.handlers:
    # Use absolute path to ensure all nodes write to the exact same file
    # no matter what their current working directory is.
    log_file_path = os.path.join(os.path.dirname(__file__), "..", "..", "agent_ros.log")
    
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    ros_logger.addHandler(file_handler)
    ros_logger.addHandler(console_handler)

class CustomRosLogger:
    def __init__(self, name):
        self._logger = ros_logger.getChild(name)
    def info(self, msg, **kwargs): self._logger.info(msg)
    def warning(self, msg, **kwargs): self._logger.warning(msg)
    def error(self, msg, **kwargs): self._logger.error(msg)
    def fatal(self, msg, **kwargs): self._logger.critical(msg)
    def debug(self, msg, **kwargs): self._logger.debug(msg)

def setup_ros_logging():
    # Monkey-patch rclpy Node to use our custom logger for cleaner terminal output
    rclpy.node.Node.get_logger = lambda self: CustomRosLogger(self.get_name())
