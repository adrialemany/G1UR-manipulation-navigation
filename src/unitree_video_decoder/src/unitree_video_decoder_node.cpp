#include <rclcpp/rclcpp.hpp>
#include <unitree_go/msg/go2_front_video_data.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

class VideoDecoder : public rclcpp::Node {
public:
    VideoDecoder() : Node("unitree_video_decoder_cpp") {
        sub_ = this->create_subscription<unitree_go::msg::Go2FrontVideoData>(
            "/frontvideostream", 10, std::bind(&VideoDecoder::callback, this, std::placeholders::_1));
        pub_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", 10);
        RCLCPP_INFO(this->get_logger(), "🚀 Decoder C++ iniciado y listo.");
    }

private:
    void callback(const unitree_go::msg::Go2FrontVideoData::SharedPtr msg) {
        if (msg->video720p.empty()) return;

        // Decodificar el buffer de video
        cv::Mat frame = cv::imdecode(cv::Mat(msg->video720p), cv::IMREAD_COLOR);
        
        if (!frame.empty()) {
            // Crear un mensaje de imagen estandar
            std_msgs::msg::Header header;
            header.stamp = this->get_clock()->now(); // Usamos la hora actual del PC
            header.frame_id = "front_camera";

            auto img_msg = cv_bridge::CvImage(header, "bgr8", frame).toImageMsg();
            pub_->publish(*img_msg);
        }
    }
    rclcpp::Subscription<unitree_go::msg::Go2FrontVideoData>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VideoDecoder>());
    rclcpp::shutdown();
    return 0;
}
