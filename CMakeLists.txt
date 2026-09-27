cmake_minimum_required(VERSION 3.20)
project(tinytrain VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3 -march=native)
endif()

# Engine library (exclude app mains)
file(GLOB_RECURSE ENGINE_SOURCES CONFIGURE_DEPENDS src/*.cpp)
list(FILTER ENGINE_SOURCES EXCLUDE REGEX ".*apps/.*\\.cpp$")
add_library(tinytrain_lib ${ENGINE_SOURCES})
target_include_directories(tinytrain_lib PUBLIC include)

# CLIs and tools
file(GLOB APP_SOURCES CONFIGURE_DEPENDS apps/*.cpp)
foreach(app_src ${APP_SOURCES})
  get_filename_component(app_name ${app_src} NAME_WE)
  add_executable(${app_name} ${app_src})
  target_link_libraries(${app_name} PRIVATE tinytrain_lib)
endforeach()

# GoogleTest via FetchContent
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# nlohmann/json (safetensors checkpoint writer)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/archive/v3.11.3.tar.gz
)
FetchContent_MakeAvailable(nlohmann_json)
target_link_libraries(tinytrain_lib PUBLIC nlohmann_json::nlohmann_json)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(tinytrain_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

enable_testing()
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)
if(TEST_SOURCES)
  add_executable(tinytrain_tests ${TEST_SOURCES})
  target_link_libraries(tinytrain_tests PRIVATE tinytrain_lib gtest_main)
  include(GoogleTest)
  gtest_discover_tests(tinytrain_tests)
endif()
