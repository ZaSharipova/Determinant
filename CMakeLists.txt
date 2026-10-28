cmake_minimum_required(VERSION 3.14)
project(multilevel_cache CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_NDEBUG OFF)
if(ENABLE_NDEBUG)
  add_compile_definitions(NDEBUG)
endif()

add_compile_options(-Wall -Wextra)

find_package(GTest REQUIRED)
enable_testing()
include(GoogleTest)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)

include_directories(${SRC_DIR})

add_executable(tests_e2e ${TEST_DIR}/e2e_tests.cpp)
target_link_libraries(tests_e2e PRIVATE GTest::gtest_main)
gtest_discover_tests(tests_e2e)

find_package(benchmark REQUIRED)
add_executable(storage_bench ${TEST_DIR}/benchmarks.cpp)
target_compile_features(storage_bench PRIVATE cxx_std_23)
target_link_libraries(storage_bench PRIVATE benchmark::benchmark)
