cmake_minimum_required(VERSION 3.10)
project(CelestialNavigation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCAMERA REQUIRED libcamera)

include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${Boost_INCLUDE_DIRS} ${LIBCAMERA_INCLUDE_DIRS})
include_directories(/home/haris/libkdtree)

add_executable(celestial_nav
    src/main.cpp
    src/image_acquisition.cpp
    src/preprocessing.cpp
    src/star_detection.cpp
    src/star_matching.cpp
    src/location_determination.cpp
    src/user_interface.cpp
)

target_link_libraries(celestial_nav
    ${OpenCV_LIBS}
    ${EIGEN3_LIBRARIES}
    ${Boost_LIBRARIES}
    ${LIBCAMERA_LIBRARIES}
)

