if(TARGET glm::glm)
    return()
endif()

message(STATUS "Third-party: creating target 'glm::glm'")

include(CPM)
CPMAddPackage("gh:g-truc/glm#2d4c4b4dd31fde06cfffad7915c2b3006402322f")

set_target_properties(glm PROPERTIES FOLDER ThirdParty)
