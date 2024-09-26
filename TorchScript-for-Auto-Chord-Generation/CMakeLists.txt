cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ChordGenerator)

set(CMAKE_PREFIX_PATH /opt/local/libtorch/share/cmake/Torch)
set(Torch_DIR /opt/local/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)