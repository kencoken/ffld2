set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/src/aux/cmake/")
include_directories("${CMAKE_SOURCE_DIR}/src")

find_package(OpenCV REQUIRED)
find_package(GLOG REQUIRED)

set (cppwrap_SOURCES
  detector/ffld_detector.cc
  util/mat_jpeg_image.cc)

set (cppwrap_LIBRARIES
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARIES})

add_library(cppwrap STATIC ${cppwrap_SOURCES})
target_link_libraries(cppwrap ${cppwrap_LIBRARIES})

# ffld2 related
target_link_libraries(cppwrap ffld2)
TARGET_LINK_LIBRARIES(cppwrap ${FFTW3_LIBRARIES})
IF(JPEG_FOUND)
  TARGET_LINK_LIBRARIES(cppwrap ${JPEG_LIBRARIES})
ENDIF()
IF(LIBXML2_FOUND)
  TARGET_LINK_LIBRARIES(cppwrap ${LIBXML2_LIBRARIES})
ENDIF()
