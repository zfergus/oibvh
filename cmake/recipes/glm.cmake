if(TARGET glm::glm)
    return()
endif()

message(STATUS "Third-party: creating target 'glm::glm'")

include(FetchContent)
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        2d4c4b4dd31fde06cfffad7915c2b3006402322f
)

FetchContent_MakeAvailable(glm)

set_target_properties(glm PROPERTIES FOLDER ThirdParty)
