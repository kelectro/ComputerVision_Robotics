cmake_minimum_required(VERSION 3.1)
project(marker2_sequence)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find external modules
find_package(OpenCV 3 REQUIRED)

# Compile options
if((CMAKE_CXX_COMPILER_ID MATCHES "Clang") OR (CMAKE_CXX_COMPILER_ID MATCHES "GNU"))
  add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wcast-qual -Wcast-align)
endif()

# Target executable
add_executable(marker2_sequence src/marker2_sequence.cpp)
target_link_libraries(marker2_sequence ${OpenCV_LIBS})
