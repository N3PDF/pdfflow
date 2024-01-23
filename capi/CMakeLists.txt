cmake_minimum_required (VERSION 3.0.2...3.28.1)

# Use rpaths for now, previously there were issues with osx
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

project(libpdfflow)

set(VERSION "\"0.1\"")

find_package(Python3 COMPONENTS Interpreter Development)

# running the cffi builder
if (NOT EXISTS ${PROJECT_SOURCE_DIR/src/cpdfflow.cc})
  execute_process(COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/src/build.py WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src)
endif()

include_directories(${Python3_INCLUDE_DIRS})
include_directories(src)
add_library(pdfflow SHARED ${PROJECT_SOURCE_DIR}/src/cpdfflow.c)
target_link_libraries(pdfflow ${Python3_LIBRARIES})

# pkg-config
set(prefix ${CMAKE_INSTALL_PREFIX})
set(exec_prefix "${prefix}")
set(includedir "${prefix}/include")
set(extraincludirs "-I${Python3_INCLUDE_DIRS}")
set(libdir "${prefix}/lib")
set(pythonlibs "${Python3_LIBRARIES}")

configure_file(
  "${PROJECT_SOURCE_DIR}/src/pdfflow.pc.in"
  "${PROJECT_SOURCE_DIR}/src/pdfflow.pc"
  )

install(FILES ${PROJECT_SOURCE_DIR}/src/pdfflow.pc DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/pkgconfig)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/pdfflow DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(TARGETS pdfflow LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
