cmake_minimum_required(VERSION 3.10)
project(clip_tokenizer_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure debug symbols and no optimization for reliable breakpoints
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g3")


include(GNUInstallDirs)
install(FILES ${PROJECT_SOURCE_DIR}/res/bpe_simple_vocab_16e6.txt.gz DESTINATION ${CMAKE_INSTALL_DATADIR}/clip_tokenizer_cpp)

include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(ZLIB REQUIRED)

add_executable(clip_tokenizer_cpp src/main.cpp src/tokenizer.cpp)

target_compile_definitions(clip_tokenizer_cpp PRIVATE
    CTC_DATA_DIR="${CMAKE_INSTALL_FULL_DATADIR}/clip_tokenizer_cpp"
)

target_link_libraries(clip_tokenizer_cpp PRIVATE ZLIB::ZLIB)

