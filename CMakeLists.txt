cmake_minimum_required(VERSION 3.8)

project("HeadAnalysis")

include(cmake/check-compiler.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /bigobj")

set(
	SOURCE_FILES 
		main.cpp
		ModelRender.cpp
		HeadAnalysis.cpp
		3rd_party/dlib/include/dlib/all/source.cpp
)

configure_file(ProjectPath.h.in ${PROJECT_BINARY_DIR}/ProjectPath.h @ONLY)

if(MSVC)
   include(cmake/add-libraries-windows.cmake)
else()
   include(cmake/add-libraries-linux.cmake)
endif()

add_executable(HeadAnalysis ${SOURCE_FILES})

if(MSVC)
   include(cmake/target-link-libraries-windows.cmake)
else()
   include(cmake/target-link-libraries-linux.cmake)
endif()

target_include_directories(HeadAnalysis PUBLIC ${CMAKE_BINARY_DIR})