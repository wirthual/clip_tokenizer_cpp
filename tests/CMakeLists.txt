# Test configuration for clip_tokenizer_cpp

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Create test executable
add_executable(test_tokenizer test_tokenizer.cpp)

# Link against the clip_tokenizer_cpp library and gtest
target_link_libraries(test_tokenizer 
    PRIVATE 
    clip_tokenizer_cpp
    GTest::gtest
    GTest::gtest_main
)

# Discover tests automatically
include(GoogleTest)
gtest_discover_tests(test_tokenizer)
