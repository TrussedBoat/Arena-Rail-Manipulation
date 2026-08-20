import os
import glob
import time
import threading
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

class GaussianSplattingNode(Node):
    def __init__(self):
        super().__init__('gaussian_splatting_node')
        
        self.declare_parameter('cache_dir', '/dev/shm/3dgs_cache')
        self.cache_dir = self.get_parameter('cache_dir').value
        
        # Service to trigger optimization
        self.srv = self.create_service(Trigger, '~/optimize_object', self.optimize_callback)
        
        # Timer to monitor the cache
        self.timer = self.create_timer(1.0, self.monitor_cache_callback)
        
        self.get_logger().info(f"Gaussian Splatting Node initialized. Monitoring {self.cache_dir}")

    def monitor_cache_callback(self):
        if not os.path.exists(self.cache_dir):
            return
            
        for object_id in os.listdir(self.cache_dir):
            obj_dir = os.path.join(self.cache_dir, object_id)
            if not os.path.isdir(obj_dir):
                continue
                
            png_files = glob.glob(os.path.join(obj_dir, '*.png'))
            num_frames = len(png_files)
            
            # For demonstration, we log if an object hits exactly 50 frames
            if num_frames == 50:
                self.get_logger().info(f"Object {object_id} has accumulated exactly {num_frames} views. Ready for optimization!")

    def optimize_callback(self, request, response):
        # We assume the object_id is passed in the request string or we just hardcode it for the skeleton
        object_id = "object_001" # In a real implementation, this would be a custom service with an object_id field
        
        obj_dir = os.path.join(self.cache_dir, object_id)
        if not os.path.exists(obj_dir):
            response.success = False
            response.message = f"Cache directory for {object_id} not found."
            return response
            
        self.get_logger().info(f"Received request to optimize {object_id}.")
        
        # Start heavy training loop in background thread
        training_thread = threading.Thread(
            target=self._run_3dgs_training, 
            args=(object_id,)
        )
        training_thread.start()
        
        response.success = True
        response.message = f"Started background training for {object_id}."
        return response

    def _run_3dgs_training(self, object_id):
        self.get_logger().info(f"[Thread] Starting 3DGS batch optimization for {object_id}...")
        
        # Simulate heavy CPU/GPU load
        time.sleep(5)
        
        self.get_logger().info(f"[Thread] Finished 3DGS optimization for {object_id}! Model saved to disk.")

def main(args=None):
    rclpy.init(args=args)
    node = GaussianSplattingNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Gaussian Splatting Node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
