cmake_minimum_required(VERSION 3.18)
project(oibvh CXX CUDA)

option(OIBVH_WITH_DEMO_APPLICATION "Build the demo application" ON)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/oibvh/")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/recipes/")

# General CMake utils
include(oibvh_cpm_cache)

# =============================================================================
# OIBVH Library
# =============================================================================

set(OIBVH_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(OIBVH_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

add_library(oibvh
    ${OIBVH_SRC_DIR}/cuda/collide.cu
    ${OIBVH_SRC_DIR}/cuda/oibvh.cu
    ${OIBVH_SRC_DIR}/cuda/oibvhTree.cu
    ${OIBVH_SRC_DIR}/cuda/scene.cu
    ${OIBVH_SRC_DIR}/cuda/transform.cu
    ${OIBVH_SRC_DIR}/utils/mesh.cpp
    ${OIBVH_SRC_DIR}/utils/shader.cpp
    ${OIBVH_SRC_DIR}/utils/utils.cpp
)

target_include_directories(oibvh PUBLIC ${OIBVH_INCLUDE_DIR})

# ------------------------------- Dependencies -------------------------------

include(glm)
target_link_libraries(oibvh PRIVATE glm::glm)

include(glad)
target_link_libraries(oibvh PRIVATE glad::glad)

find_package(CUDAToolkit)
target_link_libraries(oibvh PRIVATE CUDA::cudart)
set_target_properties(oibvh PROPERTIES CUDA_ARCHITECTURES "native")

# =============================================================================
# OIBVH Application
# =============================================================================

if(OIBVH_WITH_DEMO_APPLICATION)
    add_subdirectory(app)
endif()