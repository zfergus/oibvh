if(TARGET glad::glad)
    return()
endif()

message(STATUS "Third-party: creating target 'glad::glad'")

include(CPM)
CPMAddPackage("gh:libigl/libigl-glad#651a425101365aa6e8504988ef9bb363d066c5ee")

add_library(glad::glad ALIAS glad)

set_target_properties(glad PROPERTIES FOLDER ThirdParty)