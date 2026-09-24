cmake_minimum_required(VERSION 3.20)

project(joc_core VERSION 0.1.0 LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# Products
#   joc_core   shared library: the execution core (E-AC-3/EMDF/JOC/OAMD
#              bitstream, DSP, speaker and binaural rendering, ADM-BWF/WAV
#              writing, file task and the streaming push/pull surface)
#   joc_cli    command line frontend for file tasks
#
#   Public headers: include/joc_core.h    engine, telemetry and file task
#                   include/joc_stream.h  embedder-facing streaming surface
#
# Options
#   JOC_BUILD_TESTS     unit tests (CTest), on by default
#   JOC_BUILD_DEVTOOLS  in-tree verification tools, off: those sources live in
#                       devtools/, which is not part of the repository
#   JOC_ENABLE_AVX2     build the runtime-dispatched AVX2 kernels, on by default
#   JOC_ENABLE_AVX512   build the runtime-dispatched AVX-512 kernels, on by default
#
# Only the MSVC toolchain is validated locally; other platforms are built by CI.
# The floating-point flags of the original native library are preserved on
# purpose: /fp:precise on MSVC, -fno-fast-math elsewhere, and a static CRT so
# that no redistributable is required.
# ---------------------------------------------------------------------------

option(JOC_BUILD_TESTS "Build the unit tests" ON)
option(JOC_BUILD_DEVTOOLS "Build the in-tree verification tools (needs devtools/)" OFF)
# The DSP kernels are dispatched at run time (see src/simd/simd.h): the
# baseline units stay on the ISA every x86-64 CPU has, and these two options
# decide whether the wider units are linked in at all.  Both are on by default.
# The instruction set actually executed is chosen from CPUID/XGETBV (or the
# AArch64 baseline) when the library is first used, so a binary carrying the
# AVX-512 unit still runs on a CPU without it, and JOC_SIMD=scalar|sse2|avx2|
# avx512|neon pins one tier for verification.
option(JOC_ENABLE_AVX2 "Build the runtime-dispatched AVX2 kernels" ON)
option(JOC_ENABLE_AVX512 "Build the runtime-dispatched AVX-512 kernels" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Threads REQUIRED)

# Ninja learns MSVC's header dependencies by parsing the compiler's /showIncludes
# notes, and it only recognises the prefix it is told about.  A localized MSVC
# prints a translated prefix, and when CMake cannot detect it the notes are
# silently dropped: the build then reuses stale objects after a header changes and
# produces a binary that does not match its sources -- wrong, not just slow.  This
# only warns, because supplying the value is not always possible either: it has to
# survive the cache code page to be usable, and a build driver can compensate more
# reliably by dropping objects when a header is newer than they are.
if(MSVC AND CMAKE_GENERATOR MATCHES "Ninja" AND NOT CMAKE_CL_SHOWINCLUDES_PREFIX)
    message(WARNING
        "No /showIncludes prefix was detected, so Ninja will not track header "
        "dependencies and a header change will not rebuild what includes it.  "
        "Configure with -DCMAKE_CL_SHOWINCLUDES_PREFIX=<the text MSVC prints in "
        "front of each included file>, or make sure the compiler emits its "
        "messages in the language CMake probes for.")
endif()

# Verbatim copies of the upstream native library; see THIRD_PARTY_NOTICES.md.
# These files are never edited: they are the validated DSP kernels.
set(JOC_REUSED_SOURCES
    src/joc_core/eac3joc_core.cpp
    src/speaker/speaker_renderer.cpp
    src/binaural/binaural_renderer.cpp
)

set(JOC_INTERNAL_SOURCES
    src/adm/adm_metadata.cpp
    src/adm/adm_tracks.cpp
    src/binaural/binaural_runtime.cpp
    src/binaural/sofa_binaural_renderer.cpp
    src/eac3_transport/eac3_reader.cpp
    src/emdf/emdf_parser.cpp
    src/foundation/bit_reader.cpp
    src/foundation/fft.cpp
    src/foundation/fs_utf8.cpp
    src/foundation/mini_json.cpp
    src/foundation/sha256.cpp
    src/hrtf/jochrtf.cpp
    src/hrtf/public_filterbank.cpp
    src/hrtf/rosella_model.cpp
    src/hrtf/rosella_renderer.cpp
    src/hrtf/kernel_tables.cpp
    src/hrtf/sofa.cpp
    src/hrtf/sofa_cache.cpp
    src/hrtf/sofa_field.cpp
    src/io/adm_writer.cpp
    src/io/hdf5.cpp
    src/io/inflate.cpp
    src/io/npy.cpp
    src/io/npy_writer.cpp
    src/io/process.cpp
    src/io/wav_writer.cpp
    src/io/zip_reader.cpp
    src/joc_bitstream/joc_parser.cpp
    src/joc_core/objects16.cpp
    src/oamd/oamd_parser.cpp
    src/speaker/speaker_layout_lookup.cpp
    src/speaker/speaker_step.cpp
    src/stream/stream.cpp
    src/task/task.cpp
    src/telemetry/event_bus.cpp
    src/timeline/position_timeline.cpp
)

# ---------------------------------------------------------------------------
# Runtime-dispatched SIMD kernels (src/simd/simd.h explains the contract).
#
# One flat directory, the ISA in the file name (`kernels_intrin_<isa>.cpp`),
# never in a subdirectory.  MSVC has no function-level ISA attribute, so every
# ISA lives in its own translation unit compiled with its own flag, and
# dispatch.cpp -- built for the baseline ISA -- picks one when the library is
# first used.  Only a `kernels_intrin_*.cpp` unit ever gets a wider flag, so a
# baseline unit cannot inherit one by accident.
#
# The x86-64 baseline is SSE2 and there is no SSE2 unit on purpose: a 128-bit
# SSE2 register is the register a scalar double already occupies, so SSE2 cannot
# widen double-precision arithmetic and hand-written SSE2 would only add moves.
# AArch64 needs no probe either; ASIMD is architectural, and the NEON unit is
# how a vector path gets selected there.
# ---------------------------------------------------------------------------
set(JOC_SIMD_SOURCES
    src/simd/cpu_probe.cpp
    src/simd/dispatch.cpp
    src/simd/kernels_scalar.cpp
)
set(JOC_SIMD_DEFINES "")

set(JOC_ARCH "")
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86_64|EM64T)$")
        set(JOC_ARCH x86_64)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64|AARCH64)$")
        set(JOC_ARCH aarch64)
    endif()
endif()
if(NOT JOC_ARCH AND DEFINED CMAKE_CXX_COMPILER_ARCHITECTURE_ID)
    if(CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "x64")
        set(JOC_ARCH x86_64)
    elseif(CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "ARM64")
        set(JOC_ARCH aarch64)
    endif()
endif()

if(JOC_ARCH STREQUAL "x86_64")
    list(APPEND JOC_SIMD_DEFINES JOC_SIMD_HAVE_SSE2=1)
    if(JOC_ENABLE_AVX2)
        list(APPEND JOC_SIMD_SOURCES src/simd/kernels_intrin_avx2.cpp)
        list(APPEND JOC_SIMD_DEFINES JOC_SIMD_HAVE_AVX2=1)
        if(MSVC)
            set_source_files_properties(src/simd/kernels_intrin_avx2.cpp
                PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
        else()
            set_source_files_properties(src/simd/kernels_intrin_avx2.cpp
                PROPERTIES COMPILE_OPTIONS "-mavx2")
        endif()
    endif()
    if(JOC_ENABLE_AVX512)
        list(APPEND JOC_SIMD_SOURCES src/simd/kernels_intrin_avx512.cpp)
        list(APPEND JOC_SIMD_DEFINES JOC_SIMD_HAVE_AVX512=1)
        if(MSVC)
            set_source_files_properties(src/simd/kernels_intrin_avx512.cpp
                PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
        else()
            set_source_files_properties(src/simd/kernels_intrin_avx512.cpp
                PROPERTIES COMPILE_OPTIONS "-mavx512f")
        endif()
    endif()
elseif(JOC_ARCH STREQUAL "aarch64")
    list(APPEND JOC_SIMD_DEFINES JOC_SIMD_HAVE_NEON=1)
    list(APPEND JOC_SIMD_SOURCES src/simd/kernels_intrin_neon.cpp)
endif()

add_library(joc_simd OBJECT ${JOC_SIMD_SOURCES})
target_include_directories(joc_simd PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_definitions(joc_simd PRIVATE ${JOC_SIMD_DEFINES})
# The objects are linked into the shared library as well, so they must be
# position independent even though an object library does not inherit that.
set_target_properties(joc_simd PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Static form of the engine, used by the in-tree tools and tests so they can use
# internal modules without exporting them from the shared library.
if(JOC_BUILD_TESTS OR JOC_BUILD_DEVTOOLS)
    add_library(joc_core_impl STATIC ${JOC_INTERNAL_SOURCES} ${JOC_REUSED_SOURCES}
        $<TARGET_OBJECTS:joc_simd>)
    target_include_directories(joc_core_impl
        PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    target_link_libraries(joc_core_impl PUBLIC Threads::Threads)
endif()

add_library(joc_core SHARED
    src/api/joc_api.cpp
    src/api/joc_stream_api.cpp
    src/api/joc_task_api.cpp
    ${JOC_INTERNAL_SOURCES}
    ${JOC_REUSED_SOURCES}
    $<TARGET_OBJECTS:joc_simd>
)
target_include_directories(joc_core
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_compile_definitions(joc_core PRIVATE JOC_BUILD_DLL)
target_link_libraries(joc_core PRIVATE Threads::Threads)
set_target_properties(joc_core PROPERTIES
    OUTPUT_NAME "joc_core"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    POSITION_INDEPENDENT_CODE YES
)

# The frontend compiles the UTF-8 path shim itself: it is small, and the shared
# library keeps its internals unexported.
add_executable(joc_cli src/cli/joc_cli.cpp src/foundation/fs_utf8.cpp)
target_link_libraries(joc_cli PRIVATE joc_core)
target_include_directories(joc_cli PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

# The installed CLI is in bin/ and the shared library in lib/, and ELF/Mach-O
# strip the build rpath on install, so the relative lookup has to be recorded
# here or `bin/joc_cli` cannot find `../lib/libjoc_core.*`.
if(UNIX)
    if(APPLE)
        set_target_properties(joc_cli PROPERTIES INSTALL_RPATH "@loader_path/../lib")
    else()
        set_target_properties(joc_cli PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
    endif()
endif()

if(JOC_BUILD_TESTS)
    enable_testing()
    add_executable(joc_tests tests/test_core.cpp)
    target_link_libraries(joc_tests PRIVATE joc_core_impl)
    target_include_directories(joc_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME core COMMAND joc_tests)
    # The public headers must stay valid C and C++: these targets exist to prove it.
    add_library(joc_headers_c OBJECT tests/test_headers.c)
    target_include_directories(joc_headers_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    add_library(joc_headers_cpp OBJECT tests/test_headers.cpp)
    target_link_libraries(joc_headers_cpp PRIVATE joc_core)
endif()

if(JOC_BUILD_DEVTOOLS)
    add_executable(joc_dump devtools/joc_dump/main.cpp)
    target_link_libraries(joc_dump PRIVATE joc_core joc_core_impl)
    target_include_directories(joc_dump PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    # SOFA reader probe: the C++ side of devtools/checks/check_sofa.py.
    add_executable(joc_sofa_field_probe devtools/sofa_field_probe/main.cpp)
    add_executable(joc_dictionary_probe devtools/sofa_field_probe/dictionary_main.cpp)
    target_link_libraries(joc_dictionary_probe PRIVATE joc_core_impl)
    target_include_directories(joc_dictionary_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(joc_sofa_field_probe PRIVATE joc_core_impl)
    target_include_directories(joc_sofa_field_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    # Rosella model probe: the C++ side of devtools/checks/check_rosella.py.
    add_executable(joc_rosella_probe devtools/rosella_probe/main.cpp)
    target_link_libraries(joc_rosella_probe PRIVATE joc_core_impl)
    target_include_directories(joc_rosella_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_executable(joc_sofa_probe devtools/sofa_probe/main.cpp)
    target_link_libraries(joc_sofa_probe PRIVATE joc_core_impl)
    target_include_directories(joc_sofa_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    # Probe for the streaming surface: the acceptance harness and the usage
    # example for joc_stream.h.  Not shipped: a player integrates the library.
    add_executable(joc_stream_probe devtools/joc_stream/main.cpp)
    target_link_libraries(joc_stream_probe PRIVATE joc_core)
endif()

set(JOC_MSVC_TARGETS joc_core joc_cli joc_simd)
if(JOC_BUILD_TESTS OR JOC_BUILD_DEVTOOLS)
    list(APPEND JOC_MSVC_TARGETS joc_core_impl)
endif()
if(JOC_BUILD_TESTS)
    list(APPEND JOC_MSVC_TARGETS joc_tests joc_headers_c joc_headers_cpp)
endif()
if(JOC_BUILD_DEVTOOLS)
    list(APPEND JOC_MSVC_TARGETS joc_rosella_probe joc_dump joc_stream_probe joc_sofa_probe joc_sofa_field_probe joc_dictionary_probe)
endif()

if(MSVC)
    set(JOC_MSVC_FLAGS /W4 /permissive- /Zc:__cplusplus /utf-8 /EHsc /GR- /fp:precise)
    # The writers use std::fopen for seekable, byte-exact output.
    set(JOC_MSVC_DEFINES _CRT_SECURE_NO_WARNINGS)
    # C4324 comes from the reused kernel's std::barrier member at /W4; it is
    # pre-existing behaviour of a verbatim file, so the warning is silenced
    # rather than the file edited.
    set_source_files_properties(src/joc_core/eac3joc_core.cpp
        PROPERTIES COMPILE_OPTIONS "/wd4324")
    foreach(target IN LISTS JOC_MSVC_TARGETS)
        set_property(TARGET ${target} PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        # No /arch here on purpose: the baseline units keep the architecture's
        # guaranteed ISA and only the src/simd `kernels_intrin_*` units carry a
        # wider one (their flags are set per source file above).
        target_compile_options(${target} PRIVATE ${JOC_MSVC_FLAGS}
            $<$<CONFIG:Release>:/O2>
            $<$<CONFIG:Release>:/Oi>)
        target_compile_definitions(${target} PRIVATE ${JOC_MSVC_DEFINES})
    endforeach()
    target_link_options(joc_core PRIVATE /INCREMENTAL:NO /OPT:REF /OPT:ICF)
    target_link_options(joc_cli PRIVATE /INCREMENTAL:NO)
else()
    foreach(target IN LISTS JOC_MSVC_TARGETS)
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math
            $<$<CONFIG:Release>:-O3>)
    endforeach()
endif()

# Headers listed for IDE visibility.
target_sources(joc_core PRIVATE
    include/joc_core.h
    include/joc_stream.h
    include/eac3joc_core.h
    src/joc_bitstream/joc_huffman_tables.h
    src/joc_core/qmf_tables.h
    src/speaker/speaker_layouts.h
)

install(TARGETS joc_core joc_cli
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
