cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(SceneNetv1.0)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

SET(CMAKE_CXX_FLAGS "-std=c++11")

FIND_PACKAGE(CUDA REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(Pangolin REQUIRED)
find_package(CVD REQUIRED)
find_package(MinimalImgUtilities REQUIRED)
find_package(OpenMP)
find_package(Boost COMPONENTS thread REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenCV2 REQUIRED)
find_package(SceneGraph REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


# Adapt it accordingly
SET(CUDA_NVCC_FLAGS  "-arch=sm_30"  "--use_fast_math" "-O3" "-lineinfo")

#FIND_PATH(CUDA_CUT_INCLUDE_DIR
#   cutil.h
#   PATHS ${CUDA_SDK_SEARCH_PATH}
#   PATH_SUFFIXES "common/inc"
#   DOC "Location of cutil.h"
#   NO_DEFAULT_PATH
#)


#message("CUDA_SDK_SEARCH_PATH = ${CUDA_SDK_SEARCH_PATH}")

link_directories( ${GTKMM_LIBRARY_DIRS})
include_directories( ${GTKMM_INCLUDE_DIRS})

INCLUDE_DIRECTORIES(
            ${Pangolin_INCLUDE_DIR}
            ${CG_INCLUDE_PATH}
            ${EIGEN3_INCLUDE_DIR}
            ${CVD_INCLUDE_DIR}
            ${OPENGL_INCLUDE_DIR}
	    ${MinimalImgUtilities_INCLUDE_DIR}
            ${OpenCV2_INCLUDE_PATH}
            ${OPENCV_INCLUDE_DIRS}
            ${CUDA_INCLUDE_DIRS}
            ${SceneGraph_INCLUDE_DIRS}
	    /usr/lib/x86_64-linux-gnu/glibmm-2.4/include/
            /usr/include/glib-2.0/
	    /usr/lib/x86_64-linux-gnu/glib-2.0/include/
	    /usr/include/glibmm-2.4
            )

LINK_LIBRARIES(
            ${Pangolin_LIBRARY}
            ${Eigen3_LIBRARY}
            ${CVD_LIBRARY}
            ${OpenCV_LIBS}
	    ${MinimalImgUtilities_LIBRARY}
            ${SceneGraph_LIBRARIES}
            -lGLEW
            -lglut
            -lGLU
            -lGL
            -lboost_system
            -lboost_filesystem
            -lnoise
            )

LINK_DIRECTORIES(
  ${OPENCV_LIBRARY_DIRS}
  ${SceneGraph_LIBRARY_DIRS}
  /usr/local/lib
  /usr/local/lib64
)

SET(TINYOBJLOADER_SOURCES
            ./utils/map_object_label2training_label.h
            ./tinyobjloader/tiny_obj_loader.h
            ./tinyobjloader/tiny_obj_loader.cc
            ./tinyobjloader/obj_writer.h
            ./tinyobjloader/obj_writer.cc
            )


SET(VaFRIC_SOURCES
    ./VaFRIC/VaFRIC.h
    ./VaFRIC/VaFRIC.cpp)

SET(NOISE_SOURCES
    ./utils/noise/noise.h
    ./utils/noise/gaussian.h
    ./utils/noise/noiseutils.h
    ./utils/noise/noiseutils.cpp
    ./utils/noise/perlin.h
    ./utils/noise/add_kinect_noise.h
    ./utils/noise/add_kinect_noise.cu)

SET(MATH_SOURCES
    ./math/aux_math.h
    ./math/aux_math.cu
    ./math/cumath.h
    ./math/cumath.cu
    ./math/MatUtils.h)

CUDA_ADD_EXECUTABLE(add_kinect_noise
                    ${NOISE_SOURCES}
                    ${MATH_SOURCES}
                    main_add_kinect_noise.cpp)

ADD_EXECUTABLE(opengl_depth_rendering
               ${VaFRIC_SOURCES}
               ${SHADERS_SOURCES}
               ${TINYOBJLOADER_SOURCES}
               main_depth_and_labels.cpp)

ADD_EXECUTABLE(living_room_rendering
               ${VaFRIC_SOURCES}
               ${SHADERS_SOURCES}
               ${TINYOBJLOADER_SOURCES}
               main_living_room_dynamic.cpp)


ADD_EXECUTABLE(opengl_rgbd_rendering
               main_rgbd_and_labels.cpp)

ADD_EXECUTABLE(automate_add_texture
               main_automate_add_texture.cpp)

ADD_EXECUTABLE(tum_iomanip
               main_tum_iomanip.cpp)

ADD_EXECUTABLE(ReadSceneNetTrajectories
               main_read_sceneNet_trajectories.cpp)
