cmake_minimum_required(VERSION 3.14)
project(multilevel_cache CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra)

find_package(GTest REQUIRED)
enable_testing()
include(GoogleTest)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(TEST_DIR ${SRC_DIR}/tests)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

include_directories(${INCLUDE_DIR})

add_executable(tests_e2e e2e_tests.cpp)
target_link_libraries(tests_e2e PRIVATE GTest::gtest_main)
gtest_discover_tests(tests_e2e)
