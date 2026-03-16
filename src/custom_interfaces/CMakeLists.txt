cmake_minimum_required(VERSION 3.16)
project(custom_interfaces)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(custom_interfaces
    "msg/Path.msg"
    "msg/TaskDescription.msg"
    "msg/DeliveryFeedback.msg"
    "msg/DeliveryResult.msg"
    "msg/GroundedSam.msg"
    "srv/GetGroundedSamResult.srv"
    "srv/GetTaskSequence.srv"
    "srv/GetEntranceId.srv"
    "srv/TaskRecord.srv"
    "srv/TriggerReplan.srv"
    "action/FindUnit.action"
    "action/EntranceExploration.action"
    "action/DeliveryTask.action"
    DEPENDENCIES geometry_msgs sensor_msgs
    )

ament_export_dependencies(rosidl_default_runtime)
ament_package()
