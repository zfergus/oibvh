if(TARGET glm::glm)
    return()
endif()

message(STATUS "Third-party: creating target 'glm::glm'")

include(CPM)
CPMAddPackage("gh:g-truc/glm#1.0.1")

set_target_properties(glm PROPERTIES FOLDER ThirdParty)
