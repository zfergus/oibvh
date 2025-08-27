if(TARGET glfw::glfw)
    return()
endif()

message(STATUS "Third-party: creating target 'glfw::glfw'")

include(CPM)
CPMAddPackage(
    URI "gh:glfw/glfw#3327050ca66ad34426a82c217c2d60ced61526b7"
    OPTIONS
        "GLFW_BUILD_EXAMPLES OFF"
        "GLFW_BUILD_TESTS OFF"
        "GLFW_BUILD_DOCS OFF"
        "GLFW_INSTALL OFF"
        "GLFW_VULKAN_STATIC OFF"
)

add_library(glfw::glfw ALIAS glfw)

set_target_properties(glfw PROPERTIES FOLDER ThirdParty)

# Warning config
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    target_compile_options(glfw PRIVATE
        "-Wno-missing-field-initializers"
        "-Wno-objc-multiple-method-names"
    )
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(glfw PRIVATE
        "-Wno-missing-field-initializers"
        "-Wno-objc-multiple-method-names"
    )
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(glfw PRIVATE
        "-Wno-missing-field-initializers"
        "-Wno-sign-compare"
        "-Wno-unused-parameter"
    )
endif()
