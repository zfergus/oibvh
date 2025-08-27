if(TARGET assimp::assimp)
    return()
endif()

message(STATUS "Third-party: creating target 'assimp::assimp'")

include(CPM)
CPMAddPackage("gh:assimp/assimp@6.0.2")

set_target_properties(assimp PROPERTIES FOLDER ThirdParty)