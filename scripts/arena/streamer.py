import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image

import omni.ui as ui


class TVStreamerNode(Node):
    def __init__(self):
        super().__init__('isaac_tv_streamer')
        
        # Initialize the texture provider matching your material's URI
        self.provider = ui.DynamicTextureProvider("webm_stream")
        self.latest_frame_rgba = None
        self.current_topic = ""
        self.image_sub = None
        
        # Create a subscriber to listen for the command (a string message)
        self.cmd_sub = self.create_subscription(
            String, 
            '/sim/tv/stream_command', 
            self.cmd_callback, 
            10
        )
        self.get_logger().info("TV Streamer ready! Publish a topic name to '/sim/tv/stream_command'")

    def cmd_callback(self, msg):
        requested_topic = msg.data
        
        # Ignore if we are already streaming this topic
        if requested_topic == self.current_topic:
            return 
        
        self.get_logger().info(f"Command received. Switching stream to: {requested_topic}")
        
        # Clean up the old video stream subscription if it exists
        if self.image_sub is not None:
            self.destroy_subscription(self.image_sub)
            
        # Create the new subscription based on the command
        self.image_sub = self.create_subscription(
            Image,
            requested_topic,
            self.image_callback,
            10
        )
        self.current_topic = requested_topic

    def image_callback(self, msg):
        try:
            # Decode ROS image to numpy array
            np_img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
            
            # Handle common ROS color encodings
            if msg.encoding in ['bgr8', '8UC3']:
                frame_rgba = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGBA)
            elif msg.encoding == 'rgb8':
                frame_rgba = cv2.cvtColor(np_img, cv2.COLOR_RGB2RGBA)
            else:
                frame_rgba = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGBA) # fallback
                
            self.latest_frame_rgba = frame_rgba
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def update_texture(self):
        # Push the frame to the GPU only if we have a new one
        if self.latest_frame_rgba is not None:
            height, width, _ = self.latest_frame_rgba.shape
            self.provider.set_data_array(self.latest_frame_rgba, [width, height])
            
            # Reset so we don't upload the same frame to the GPU multiple times
            self.latest_frame_rgba = None
