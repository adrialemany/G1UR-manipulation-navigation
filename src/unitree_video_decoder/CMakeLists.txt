cmake_minimum_required(VERSION 3.8)
project(unitree_video_decoder)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Buscar dependencias
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(unitree_go REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# Crear el ejecutable
add_executable(decoder_node src/unitree_video_decoder_node.cpp)

# Vincular dependencias al ejecutable
target_include_directories(decoder_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

ament_target_dependencies(decoder_node
  rclcpp
  sensor_msgs
  unitree_go
  cv_bridge
  OpenCV
)

# Instalación
install(TARGETS decoder_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
