cmake_minimum_required(VERSION 3.4...3.18)
project(libibs)

find_package(pybind11 CONFIG REQUIRED)
find_package(Qhull CONFIG REQUIRED)

pybind11_add_module(libibs MODULE src/libibs.cpp)
target_link_libraries(libibs PRIVATE Qhull::qhullcpp)
target_link_libraries(libibs PRIVATE Qhull::qhull_r)

target_compile_definitions(libibs PRIVATE VERSION_INFO=${PROJECT_VERSION})