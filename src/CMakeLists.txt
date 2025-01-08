# Note that headers are optional, and do not affect add_library, but they will not
# show up in IDEs unless they are listed in add_library.
# Optionally glob, but only for CMake 3.12 or later:
file(GLOB HEADER_LIST CONFIGURE_DEPENDS
        "${Network_SOURCE_DIR}/include/boolean_network/*.h"
        "${Network_SOURCE_DIR}/include/boolean_network/*.hpp")

file(GLOB SOURCE_LIST CONFIGURE_DEPENDS
        "${Network_SOURCE_DIR}/src/*.h"
        "${Network_SOURCE_DIR}/src/*.hpp"
        "${Network_SOURCE_DIR}/src/*.cc"
        "${Network_SOURCE_DIR}/src/*.cpp")

message("sources ${SOURCE_LIST}")
message("headers ${HEADER_LIST}")
add_library(network ${SOURCE_LIST} ${HEADER_LIST})

# We need this directory, and users of our library will need it too
target_include_directories(network PUBLIC "${Network_SOURCE_DIR}/include/")
message("headers ${Network_SOURCE_DIR}")

target_link_libraries(network PUBLIC eigen PRIVATE OpenMP::OpenMP_CXX nlohmann_json)

# IDEs should put the headers in a nice place
source_group(TREE "${PROJECT_SOURCE_DIR}/include" PREFIX "Header Files" FILES ${HEADER_LIST})