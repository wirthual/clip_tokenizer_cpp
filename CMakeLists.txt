cmake_minimum_required(VERSION 3.10)
project(clip_tokenizer_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

add_library(clip_tokenizer_cpp src/tokenizer.cpp)

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

if (CLIP_TOKENIZER_CPP_BUILD_CLI)
    add_executable(clip_tokenizer_cpp_cli src/main.cpp)
    target_link_libraries(clip_tokenizer_cpp_cli PRIVATE clip_tokenizer_cpp)
endif()

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

