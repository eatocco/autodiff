cmake_minimum_required(VERSION 3.0.0)
project(autodiff_test VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()


add_executable(autodiff_test main.cpp AutodiffWrapper.cpp)


#option(BUILD_SHARED_LIBS "Build shared libs instead of static" ON)
set(CMAKE_CXX_STANDARD ${PROJECT_VERSION})
add_subdirectory(autodiff)
add_subdirectory(toml11)
find_package(Eigen3)
target_include_directories(autodiff_test PRIVATE toml11 )
target_link_libraries(autodiff_test PRIVATE autodiff)


#add_subdirectory(Catch2)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

#set(CMAKE_PREFIX_PATH "/home/eatocco/projects/autodiff/Catch2")
include(CPack)

