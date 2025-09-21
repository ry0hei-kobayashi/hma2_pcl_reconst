#include <memory>
#include <string>
#include <limits>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include <image_transport/image_transport.hpp>
#include <image_transport/subscriber_filter.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <image_geometry/pinhole_camera_model.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>

#include "hma2_pcl_reconst/depth_traits.hpp"

namespace hma2_pcl_reconst
{

namespace enc = sensor_msgs::image_encodings;
using PointCloud = sensor_msgs::msg::PointCloud2;

using namespace std::chrono_literals;

class PointCloudXyzrgb : public rclcpp::Node
{
public:
  explicit PointCloudXyzrgb(const rclcpp::NodeOptions & options)
  : Node("pointcloud_xyzrgb", options), subscribed_(false)
  {
    this->declare_parameter("queue_size", 5);
    this->declare_parameter("exact_sync", false);
    this->declare_parameter("topic_rgb", "/head_rgbd_sensor/rgb/image_rect_color");
    this->declare_parameter("topic_depth", "/head_rgbd_sensor/depth_registered/image_rect_raw");
    //this->declare_parameter("topic_rgb", "/head_rgbd_sensor/image");
    //this->declare_parameter("topic_depth", "/head_rgbd_sensor/image");
    this->declare_parameter("use_compressed", false);

    bool use_compressed = this->get_parameter("use_compressed").as_bool();

    topic_rgb_ = this->get_parameter("topic_rgb").as_string();
    topic_depth_ = this->get_parameter("topic_depth").as_string();
    rgb_transport_ = use_compressed ? "compressed" : "raw";
    depth_transport_ = use_compressed ? "compressedDepth" : "raw";

    RCLCPP_INFO(this->get_logger(), "Configured transports: rgb=%s, depth=%s",
      rgb_transport_.c_str(), depth_transport_.c_str());

    int queue_size = this->get_parameter("queue_size").as_int();
    bool exact_sync = this->get_parameter("exact_sync").as_bool();


    rclcpp::QoS qos(10);
    pub_point_cloud_ = this->create_publisher<PointCloud>("/hma_pcl_reconst/depth_registered/points", qos);



    std::string rgb_transport = use_compressed ? "compressed" : "raw";
    std::string depth_transport = use_compressed ? "compressedDepth" : "raw";

    //sub_rgb_.subscribe(this, topic_rgb, rgb_transport, rmw_qos_profile_sensor_data);
    //sub_depth_.subscribe(this, topic_depth ,depth_transport, rmw_qos_profile_sensor_data);

    if (exact_sync) {
      exact_sync_ = std::make_shared<ExactSync>(
        ExactSyncPolicy(queue_size), sub_depth_, sub_rgb_);
      exact_sync_->registerCallback(
        std::bind(&PointCloudXyzrgb::imageCb, this, std::placeholders::_1, std::placeholders::_2));
    } else {
      sync_ = std::make_shared<Sync>(
        SyncPolicy(queue_size), sub_depth_, sub_rgb_);
      sync_->registerCallback(
        std::bind(&PointCloudXyzrgb::imageCb, this, std::placeholders::_1, std::placeholders::_2));
    }

    //camera info
    sub_info_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      "/head_rgbd_sensor/rgb/camera_info", 1,
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
        model_.fromCameraInfo(msg);
      });

    timer_ = this->create_wall_timer(100ms, [this]() {
    auto subs = pub_point_cloud_->get_subscription_count();
    
    if (subs > 0) {
      if (!subscribed_) {
        this->startSubscribing();
        RCLCPP_INFO(this->get_logger(), "New subscriber detected, subscribing to topics.");
      }
    } else {
      if (subscribed_) {
        this->stopSubscribing();
        RCLCPP_INFO(this->get_logger(), "No subscribers, unsubscribing from topics.");
        }
      }
    });

    RCLCPP_INFO(this->get_logger(), "hma2_pcl_reconst -> component initialized");
  }

private:

  std::string topic_rgb_, topic_depth_;
  std::string rgb_transport_, depth_transport_;
  rclcpp::TimerBase::SharedPtr timer_;
  bool subscribed_;
  bool use_camera_info_;

  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
  using ExactSyncPolicy = message_filters::sync_policies::ExactTime<
    sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
  using Sync = message_filters::Synchronizer<SyncPolicy>;
  using ExactSync = message_filters::Synchronizer<ExactSyncPolicy>;

  std::shared_ptr<image_transport::ImageTransport> rgb_it_, depth_it_;
  image_transport::SubscriberFilter sub_rgb_, sub_depth_;
  std::shared_ptr<Sync> sync_;
  std::shared_ptr<ExactSync> exact_sync_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_info_;
  rclcpp::Publisher<PointCloud>::SharedPtr pub_point_cloud_;

  image_geometry::PinholeCameraModel model_;

  void imageCb(const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
               const sensor_msgs::msg::Image::ConstSharedPtr & rgb_msg)
  {
    if (!model_.initialized()) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "Camera model not initialized yet.");
      return;
    }

    PointCloud::SharedPtr cloud_msg(new PointCloud);
    cloud_msg->header = depth_msg->header;
    cloud_msg->height = depth_msg->height;
    cloud_msg->width = depth_msg->width;
    cloud_msg->is_dense = false;

    sensor_msgs::PointCloud2Modifier mod(*cloud_msg);
    mod.setPointCloud2Fields(
      7,
      "x", 1, sensor_msgs::msg::PointField::FLOAT32,
      "y", 1, sensor_msgs::msg::PointField::FLOAT32,
      "z", 1, sensor_msgs::msg::PointField::FLOAT32,
      "r", 1, sensor_msgs::msg::PointField::UINT8,
      "g", 1, sensor_msgs::msg::PointField::UINT8,
      "b", 1, sensor_msgs::msg::PointField::UINT8,
      "a", 1, sensor_msgs::msg::PointField::UINT8
    );
    mod.resize(cloud_msg->width * cloud_msg->height);

    // convert depends on depth type
    if (depth_msg->encoding == enc::TYPE_16UC1) {
      convert<uint16_t>(depth_msg, rgb_msg, cloud_msg);
    } else if (depth_msg->encoding == enc::TYPE_32FC1) {
      convert<float>(depth_msg, rgb_msg, cloud_msg);
    } else {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
        "Unsupported depth encoding: %s", depth_msg->encoding.c_str());
      return;
    }

    pub_point_cloud_->publish(*cloud_msg);
  }

  void startSubscribing()
  {
    if (!subscribed_) {
      RCLCPP_INFO(this->get_logger(), "Start subscribing RGB(%s) and Depth(%s)",
                  rgb_transport_.c_str(), depth_transport_.c_str());
  
      sub_rgb_.subscribe(this, topic_rgb_, rgb_transport_, rmw_qos_profile_sensor_data);
      sub_depth_.subscribe(this, topic_depth_, depth_transport_, rmw_qos_profile_sensor_data);
  
      if (exact_sync_) {
        exact_sync_->connectInput(sub_depth_, sub_rgb_);
      } else if (sync_) {
        sync_->connectInput(sub_depth_, sub_rgb_);
      }
  
      subscribed_ = true;
    }
  }
  
  void stopSubscribing()
  {
    if (subscribed_) {
      RCLCPP_INFO(this->get_logger(), "Stop subscribing RGB and Depth");
      sub_rgb_.unsubscribe();
      sub_depth_.unsubscribe();
      subscribed_ = false;
    }
  }

  template<typename T>
  void convert(const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
               const sensor_msgs::msg::Image::ConstSharedPtr & rgb_msg,
               const PointCloud::SharedPtr & cloud_msg)
  {
    float center_x = model_.cx();
    float center_y = model_.cy();
    double unit_scaling = hma2_pcl_reconst::DepthTraits<T>::toMeters(T(1));
    float constant_x = unit_scaling / model_.fx();
    float constant_y = unit_scaling / model_.fy();
    float bad_point = std::numeric_limits<float>::quiet_NaN();

    const T * depth_row = reinterpret_cast<const T *>(&depth_msg->data[0]);
    int row_step = depth_msg->step / sizeof(T);
    const uint8_t * rgb = &rgb_msg->data[0];
    int color_step = 3;  // RGB8

    sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(*cloud_msg, "r");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(*cloud_msg, "g");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(*cloud_msg, "b");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_a(*cloud_msg, "a");

    for (int v = 0; v < static_cast<int>(cloud_msg->height); ++v, depth_row += row_step) {
      for (int u = 0; u < static_cast<int>(cloud_msg->width);
           ++u, rgb += color_step,
           ++iter_x, ++iter_y, ++iter_z, ++iter_r, ++iter_g, ++iter_b, ++iter_a)
      {
        T depth = depth_row[u];
        if (!hma2_pcl_reconst::DepthTraits<T>::valid(depth)) {
          *iter_x = *iter_y = *iter_z = bad_point;
        } else {
          *iter_x = (u - center_x) * depth * constant_x;
          *iter_y = (v - center_y) * depth * constant_y;
          *iter_z = hma2_pcl_reconst::DepthTraits<T>::toMeters(depth);
        }
        *iter_r = rgb[0];
        *iter_g = rgb[1];
        *iter_b = rgb[2];
        *iter_a = 255;
      }
    }
  }
};

} // namespace hma2_pcl_reconst

RCLCPP_COMPONENTS_REGISTER_NODE(hma2_pcl_reconst::PointCloudXyzrgb)

