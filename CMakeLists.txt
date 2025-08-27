cmake_minimum_required(VERSION 3.18)
project(oibvh C CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/oibvh/")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/find/")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/recipes/")

# General CMake utils
include(oibvh_cpm_cache)

################################################################################
# OIBVH Application
################################################################################

set(oibvh_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(oibvh_INCLUDE_DIR ${oibvh_BASE_DIR}/include)
set(oibvh_SRC_DIR ${oibvh_BASE_DIR}/src)

aux_source_directory(${oibvh_SRC_DIR} oibvh_SRC)
aux_source_directory(${oibvh_SRC_DIR}/utils oibvh_SRC)
aux_source_directory(${oibvh_SRC_DIR}/cuda oibvh_SRC)
aux_source_directory(${oibvh_SRC_DIR}/cpu oibvh_SRC)

add_executable(oibvh ${oibvh_SRC})

target_include_directories(oibvh PUBLIC ${oibvh_INCLUDE_DIR})

target_compile_definitions(oibvh PRIVATE OIBVH_SHADER_DIR="${oibvh_BASE_DIR}/shaders")
target_compile_definitions(oibvh PRIVATE OIBVH_OBJECT_DIR="${oibvh_BASE_DIR}/objects")

################################################################################
# Dependencies
################################################################################

include(glfw)
target_link_libraries(oibvh PRIVATE glfw::glfw)

include(glad)
target_link_libraries(oibvh PRIVATE glad::glad)

include(imgui)
target_link_libraries(oibvh PRIVATE imgui::imgui)

include(stb)
target_link_libraries(oibvh PRIVATE stb::stb)

include(glm)
target_link_libraries(oibvh PRIVATE glm::glm)

include(assimp)
target_link_libraries(oibvh PRIVATE assimp::assimp)

find_package(CUDAToolkit)
target_link_libraries(oibvh PRIVATE CUDA::cudart)
set_target_properties(oibvh PROPERTIES CUDA_ARCHITECTURES "native")