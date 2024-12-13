cmake_minimum_required(VERSION 3.22)
project(aircraft_route LANGUAGES CXX CUDA)

# Set C++ standard version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard version
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compiler flags for C++
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -m64 -fopenmp -Wno-unknown-pragmas")

# Set the CUDA compiler
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/bin/nvcc)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 61)

# Set CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --gpu-architecture=compute_61")

# Define the source files
set(SOURCE_FILES
        Cuda/aircraft_route.cu
        OpenMP/flightroute.cc
)

# Add the executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Link CUDA libraries
set(CUDA_LIB_DIR /usr/local/cuda-11.7/lib64/)
link_directories(${CUDA_LIB_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE cudart)

# Enable output directory for objects
set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
