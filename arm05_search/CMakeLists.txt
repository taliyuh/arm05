cmake_minimum_required(VERSION 3.8)
project(arm05_search)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_python_install_package(bt_scripts)

install(PROGRAMS
  bt_scripts/search_server.py
  bt_scripts/aruco_detector.py
  bt_scripts/evaluate_client.py
  DESTINATION lib/${PROJECT_NAME}/
)

ament_package()
