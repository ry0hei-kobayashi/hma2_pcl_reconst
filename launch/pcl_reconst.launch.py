from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    container = ComposableNodeContainer(
        name="cloud_reconst_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[
            ComposableNode(
                package="hma_pcl_reconst2",
                plugin="hma_pcl_reconst2::PointCloudXyzrgb",
                name="pcl_reconst",
                parameters=[{
                    "queue_size": 5,
                    "exact_sync": False,
                    "use_compressed": False,
                }]
            )
        ],
        output="screen"
    )
    return LaunchDescription([container])

