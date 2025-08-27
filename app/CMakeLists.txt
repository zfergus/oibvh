add_executable(oibvh_demo
    src/main.cpp
    src/cpu/simpleBVH.cpp
    src/cpu/simpleCollide.cpp
    src/utils/camera.cpp
    src/utils/drawableOibvhTree.cu
    src/utils/renderMesh.cpp
    src/utils/model.cpp
    src/utils/scene.cu
    src/utils/transform.cu
    src/utils/shader.cpp
)

target_include_directories(oibvh_demo PUBLIC ${oibvh_INCLUDE_DIR})

target_compile_definitions(oibvh_demo PRIVATE OIBVH_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders")
target_compile_definitions(oibvh_demo PRIVATE OIBVH_OBJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/objects")

set_target_properties(oibvh_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    OUTPUT_NAME "oibvh"
)

# ------------------------- Dependencies ----------------------------------

target_link_libraries(oibvh_demo PRIVATE oibvh)

include(glfw)
target_link_libraries(oibvh_demo PRIVATE glfw::glfw)

include(glad)
target_link_libraries(oibvh_demo PRIVATE glad::glad)

include(imgui)
target_link_libraries(oibvh_demo PRIVATE imgui::imgui)

include(stb)
target_link_libraries(oibvh_demo PRIVATE stb::stb)

include(glm)
target_link_libraries(oibvh_demo PRIVATE glm::glm)

include(assimp)
target_link_libraries(oibvh_demo PRIVATE assimp::assimp)

find_package(CUDAToolkit)
target_link_libraries(oibvh_demo PRIVATE CUDA::cudart)
set_target_properties(oibvh_demo PROPERTIES CUDA_ARCHITECTURES "native")