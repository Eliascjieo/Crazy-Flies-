cmake_minimum_required(VERSION 3.15)
project(optitrack_client)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)

set(NATNET_SDK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/NatNetSDK)

add_executable(${PROJECT_NAME}
  src/optitrack_client.cpp
  src/utils.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${NATNET_SDK_DIR}/include)

target_link_directories(${PROJECT_NAME} PUBLIC
  ${NATNET_SDK_DIR}/lib)

target_link_libraries(${PROJECT_NAME} NatNet)

ament_target_dependencies(${PROJECT_NAME} rclcpp geometry_msgs)

install(TARGETS ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY ${NATNET_SDK_DIR}/lib/
  DESTINATION lib)

ament_package()
