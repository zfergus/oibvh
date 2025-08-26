if(TARGET assimp::assimp)
    return()
endif()

find_package(assimp REQUIRED)

add_library(assimp INTERFACE IMPORTED)
add_library(assimp::assimp ALIAS assimp)

target_include_directories(assimp INTERFACE ${ASSIMP_INCLUDE_DIRS})
target_link_libraries(assimp INTERFACE ${ASSIMP_LIBRARIES})