cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(openni_grabber)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (openni_grabber openni_grabber.cpp)
target_link_libraries (openni_grabber ${PCL_LIBRARIES})

find_package(PNGwriter 0.7.0)

if(PNGwriter_FOUND)
  target_link_libraries(openni_grabber PRIVATE PNGwriter::PNGwriter)
endif(PNGwriter_FOUND)

message(PCL_DIR ${PCL_DIR})
