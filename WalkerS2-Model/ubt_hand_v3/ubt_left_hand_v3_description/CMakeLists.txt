cmake_minimum_required(VERSION 3.10)
project(ubt_left_hand_v3_description)

# Default to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-Wall -Werror -Wno-enum-compare)

find_package(ament_cmake REQUIRED)
find_package(urdf REQUIRED)
if("$ENV{ROSA_VERSION}" STREQUAL "2")
  find_package(rosa_generate_tools REQUIRED)
else()

endif()
# ##############################################################################
# Install ##
# ##############################################################################
install(DIRECTORY meshes DESTINATION share/${PROJECT_NAME})
install(DIRECTORY urdf DESTINATION share/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_package()
