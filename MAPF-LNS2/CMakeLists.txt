cmake_minimum_required(VERSION 3.5...3.26)
project(solver LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 14)
set(PYBIND11_PYTHON_VERSION 3.6)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC")

include_directories("inc" "inc/CBS" "inc/PIBT")

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
find_package(Eigen3 REQUIRED)

file(GLOB SOURCES "src/*.cpp" "src/CBS/*.cpp" "src/PIBT/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX ".*src/driver.cpp")
list(FILTER SOURCES EXCLUDE REGEX ".*src/testmain.cpp")
# message(SOURCES="${SOURCES}")

# pybind11_add_module(${PROJECT_NAME} npinput.cpp)
pybind11_add_module(${PROJECT_NAME}  ${SOURCES})

target_compile_definitions(${PROJECT_NAME} PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_include_directories(${PROJECT_NAME} PRIVATE ${PYBIND11_INCLUDE_DIRS} /usr/include/eigen3 .)
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)