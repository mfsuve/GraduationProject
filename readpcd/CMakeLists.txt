cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(readpcd)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (readpcd readpcd.cpp)
target_link_libraries (readpcd ${PCL_LIBRARIES})

find_package(PNGwriter 0.7.0)

if(PNGwriter_FOUND)
  target_link_libraries(readpcd PRIVATE PNGwriter::PNGwriter)
endif(PNGwriter_FOUND)
