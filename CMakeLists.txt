cmake_minimum_required(VERSION 3.10)
project(clip_tokenizer_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable testing
enable_testing()

# Ensure debug symbols and no optimization for reliable breakpoints
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g3")


include(GNUInstallDirs)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(CLIP_TOKENIZER_CPP_BUILD_CLI_DEFAULT ON)
else()
    set(CLIP_TOKENIZER_CPP_BUILD_CLI_DEFAULT OFF)
endif()

option(CLIP_TOKENIZER_CPP_BUILD_CLI "Build the CLI example" ${CLIP_TOKENIZER_CPP_BUILD_CLI_DEFAULT})

find_package(ZLIB REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG QUIET)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED libpcre2-8)

add_subdirectory(extern/pybind)

add_library(clip_tokenizer_cpp src/tokenizer.cpp)

# Enable Position Independent Code for use in shared libraries
set_target_properties(clip_tokenizer_cpp PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(clip_tokenizer_cpp
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(clip_tokenizer_cpp
    PUBLIC
    $<BUILD_INTERFACE:CTC_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/res\">
        $<INSTALL_INTERFACE:CTC_DATA_DIR=\"${CMAKE_INSTALL_FULL_DATADIR}/clip_tokenizer_cpp\">
)

target_link_libraries(clip_tokenizer_cpp PUBLIC ZLIB::ZLIB)
target_include_directories(clip_tokenizer_cpp PUBLIC ${PCRE2_INCLUDE_DIRS})
target_compile_options(clip_tokenizer_cpp PUBLIC ${PCRE2_CFLAGS_OTHER})
target_link_libraries(clip_tokenizer_cpp PUBLIC ${PCRE2_LIBRARIES})

if (CLIP_TOKENIZER_CPP_BUILD_CLI)
    add_executable(clip_tokenizer_cpp_cli src/main.cpp)
    target_link_libraries(clip_tokenizer_cpp_cli PRIVATE clip_tokenizer_cpp)
endif()

# Build pybind11 bindings
pybind11_add_module(clip_tokenizer_cpp_py src/bindings.cpp)
target_link_libraries(clip_tokenizer_cpp_py PRIVATE clip_tokenizer_cpp)

# Optional: Install the Python module
install(TARGETS clip_tokenizer_cpp_py
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python
)

install(TARGETS clip_tokenizer_cpp
    EXPORT clip_tokenizer_cppTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/res/bpe_simple_vocab_16e6.txt.gz
    DESTINATION ${CMAKE_INSTALL_DATADIR}/clip_tokenizer_cpp
)

install(EXPORT clip_tokenizer_cppTargets
    NAMESPACE clip_tokenizer_cpp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/clip_tokenizer_cpp
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/clip_tokenizer_cppConfigVersion.cmake
    VERSION 1.0.0
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/clip_tokenizer_cppConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/clip_tokenizer_cppConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/clip_tokenizer_cpp
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/clip_tokenizer_cppConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/clip_tokenizer_cppConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/clip_tokenizer_cpp
)

# Add tests subdirectory
message(STATUS "Adding tests subdirectory...")
add_subdirectory(tests)
message(STATUS "Tests subdirectory added")


