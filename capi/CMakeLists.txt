cmake_minimum_required (VERSION 3.0.2)

# Use rpaths for now, previously there were issues with osx
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

project(libpdfflow)

set(VERSION "\"0.1\"")

find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs   3 REQUIRED)

# running the cffi builder
if (NOT EXISTS ${PROJECT_SOURCE_DIR/src/cpdfflow.cc})
  execute_process(COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/src/build.py WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src)
endif()

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(src)
add_library(pdfflow SHARED ${PROJECT_SOURCE_DIR}/src/cpdfflow.cc)

# pkg-config
set(prefix ${CMAKE_INSTALL_PREFIX})
set(exec_prefix "${prefix}")
set(includedir "${prefix}/include")
set(extraincludirs "-I${PYTHON_INCLUDE_DIR}")
set(libdir "${prefix}/lib")
set(pythonlibs "${PYTHON_LIBRARIES}")

configure_file(
  "${PROJECT_SOURCE_DIR}/src/pdfflow.pc.in"
  "${PROJECT_SOURCE_DIR}/src/pdfflow.pc"
  )

install(FILES ${PROJECT_SOURCE_DIR}/src/pdfflow.pc DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/pkgconfig)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/pdfflow DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(TARGETS pdfflow LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)