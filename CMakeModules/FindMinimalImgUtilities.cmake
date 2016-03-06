# - Try to find libMinimalImgUtilities
#
#  MinimalImgUtilities_FOUND - system has libMinimalImgUtilities
#  MinimalImgUtilities_INCLUDE_DIR - the libMinimalImgUtilities include directories
#  MinimalImgUtilities_LIBRARY - link these to use libMinimalImgUtilities


FIND_PATH(
  MinimalImgUtilities_INCLUDE_DIR_SRC
  NAMES iucore.h
  PATHS
    ${CMAKE_SOURCE_DIR}/../imageutilities/src
    ${CMAKE_SOURCE_DIR}/../imageutilities/src
    /usr/include
    /usr/local/include
)

SET( MinimalImgUtilities_INCLUDE_DIR_BUILD ${CMAKE_SOURCE_DIR}/../imageutilities/build/ )

#MESSAGE("MinimalImgUtilities_INCLUDE_DIR_SRC " ${MinimalImgUtilities_INCLUDE_DIR_SRC})
#MESSAGE("MinimalImgUtilities_INCLUDE_DIR_BUILD " ${MinimalImgUtilities_INCLUDE_DIR_BUILD})
#MESSAGE("Find COmponent "${MinimalImgUtilities_FIND_COMPONENTS})

set(IU_MODULES iucore iuio iuiopgm)

set(MinimalImgUtilities_LIBRARY ${CMAKE_SOURCE_DIR}/../imageutilities/build/src)

list(APPEND IU_LIBS "")

FOREACH(IU_MODULE ${IU_MODULES})
FIND_LIBRARY(
  MinimalImgUtilities_LIBRARY_${IU_MODULE}
  NAMES ${IU_MODULE}
  PATHS
    ${CMAKE_SOURCE_DIR}/../imageutilities/build/src/
    /usr/lib
    /usr/local/lib
) 

#IF(NOT MinimalImgUtilities_LIBRARY_${IU_MODULE})
#	set (MinimalImgUtilities_LIBRARY " ")
#ELSE
IF(MinimalImgUtilities_LIBRARY_${IU_MODULE})
	list(APPEND IU_LIBS ${MinimalImgUtilities_LIBRARY_${IU_MODULE}})

ENDIF(MinimalImgUtilities_LIBRARY_${IU_MODULE})

#MESSAGE("Found Library: " ${MinimalImgUtilities_LIBRARY_${IU_MODULE}})

ENDFOREACH(IU_MODULE)

IF(MinimalImgUtilities_LIBRARY)
	set(MinimalImgUtilities_LIBRARY ${IU_LIBS})
        #MESSAGE("MinimalImgUtilities_LIBRARY " ${MinimalImgUtilities_LIBRARY})
ENDIF(MinimalImgUtilities_LIBRARY)

IF(MinimalImgUtilities_INCLUDE_DIR_SRC AND MinimalImgUtilities_INCLUDE_DIR_BUILD AND MinimalImgUtilities_LIBRARY)
  SET(MinimalImgUtilities_INCLUDE_DIR ${MinimalImgUtilities_INCLUDE_DIR_SRC};${MinimalImgUtilities_INCLUDE_DIR_BUILD})
  SET(MinimalImgUtilities_FOUND TRUE)
ENDIF()

#MESSAGE("CMAKE SOURCE DIRECTORY = ${CMAKE_SOURCE_DIR}")

IF(MinimalImgUtilities_FOUND)
   IF(NOT MinimalImgUtilities_FIND_QUIETLY)
      MESSAGE(STATUS "Found MinimalImgUtilities: ${MinimalImgUtilities_LIBRARY}")
   ENDIF()
ELSE()
   IF(MinimalImgUtilities_FIND_REQUIRED)
      MESSAGE(FATAL_ERROR "Could not find MinimalImgUtilities")
   ENDIF()
ENDIF()


