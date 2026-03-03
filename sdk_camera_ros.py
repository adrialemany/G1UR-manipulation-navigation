import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

# Importaciones del Unitree SDK2
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient

class UnitreeSDKCameraFast(Node):
    def __init__(self):
        super().__init__('unitree_sdk_camera_fast')
        # Ahora publicamos en un topic comprimido
        self.publisher_ = self.create_publisher(CompressedImage, '/camera/image/compressed', 10)
        
        ChannelFactoryInitialize(0, "eth0") 
        self.video_client = VideoClient()
        self.video_client.SetTimeout(3.0)
        self.video_client.Init()
        
        self.timer = self.create_timer(0.04, self.timer_callback)
        self.get_logger().info("🚀 Mandando vídeo COMPRIMIDO a la red (Adiós lag)...")

    def timer_callback(self):
        code, data = self.video_client.GetImageSample()
        if code == 0 and data is not None:
            # Creamos el mensaje comprimido
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_face"
            msg.format = "jpeg"
            
            # ¡Magia! Metemos los bytes del robot directamente en la red sin convertirlos
            msg.data = bytes(data) 
            
            self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = UnitreeSDKCameraFast()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
