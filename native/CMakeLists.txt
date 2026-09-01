cmake_minimum_required(VERSION 3.20)

project(eac3joc_core VERSION 1.0.0 LANGUAGES CXX)

include(GNUInstallDirs)
find_package(Threads REQUIRED)

add_library(eac3joc_core SHARED
    src/eac3joc_core.cpp
    src/speaker_renderer.cpp
    src/joc_huffman_tables.h
    src/qmf_tables.h
    src/speaker_layouts.h
)

target_compile_features(eac3joc_core PRIVATE cxx_std_20)
target_include_directories(eac3joc_core PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(eac3joc_core PRIVATE Threads::Threads)

set_target_properties(eac3joc_core PROPERTIES
    OUTPUT_NAME "eac3joc_core"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    POSITION_INDEPENDENT_CODE YES
)

if(MSVC)
    set_property(TARGET eac3joc_core PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    target_compile_options(eac3joc_core PRIVATE
        /W4 /permissive- /Zc:__cplusplus /utf-8 /EHsc /GR- /fp:precise
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/Oi>
        $<$<CONFIG:Release>:/GL>
    )
    target_link_options(eac3joc_core PRIVATE
        $<$<CONFIG:Release>:/LTCG>
        /INCREMENTAL:NO /OPT:REF /OPT:ICF
    )
else()
    target_compile_options(eac3joc_core PRIVATE
        -Wall -Wextra -Wpedantic -fno-fast-math
        $<$<CONFIG:Release>:-O3>
    )
endif()

# Install directly into the chosen prefix. Recommended invocation from the
# repository root:
#   cmake -S native -B build/cmake -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_INSTALL_PREFIX=<repo>/lib
#   cmake --build build/cmake --config Release
#   cmake --install build/cmake --config Release
#
# Result names are supplied by the platform toolchain:
#   Windows: eac3joc_core.dll (+ eac3joc_core.lib import library)
#   Linux:   libeac3joc_core.so
#   macOS:   libeac3joc_core.dylib
install(TARGETS eac3joc_core
    RUNTIME DESTINATION .
    LIBRARY DESTINATION .
    ARCHIVE DESTINATION .
)

if(MSVC)
    install(FILES "$<TARGET_PDB_FILE:eac3joc_core>"
        DESTINATION .
        OPTIONAL
    )
endif()

