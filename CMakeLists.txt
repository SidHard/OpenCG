###################    OpenCG    ########################

cmake_minimum_required(VERSION 2.8)
project(OpenCG)

find_package(OpenCV REQUIRED)
find_package(CUDA)

OPTION(WITH_CUDA "CUDA support?" ${CUDA_FOUND})

IF(MSVC_IDE)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
ENDIF(MSVC_IDE)


add_subdirectory(CGLib)

set(EXECUTABLE_OUTPUT_PATH, ${PROJECT_SOURCE_DIR}/bin)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/CGLib)
IF(WITH_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  cuda_add_executable(OpenCG OpenCG.cpp)
  target_link_libraries(OpenCG ${CUDA_LIBRARY})
ELSE()
  add_definitions(-DCOMPILE_WITHOUT_CUDA)
  add_executable(OpenCG OpenCG.cpp)
ENDIF()

target_link_libraries(OpenCG ${OpenCV_LIBS})
target_link_libraries(OpenCG CGLib)

