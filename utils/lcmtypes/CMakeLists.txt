   
set(LIBRARY_OUTPUT_PATH)

FILE(GLOB lcm_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} [^.]*.lcm)

foreach(lcm_file ${lcm_files})
   
    STRING(REGEX REPLACE "t.lcm" "" file_stub ${lcm_file})
    STRING(REGEX REPLACE "t.lcm" "t.c" c_file ${lcm_file})
    STRING(REGEX REPLACE "t.lcm" "t.h" h_file ${lcm_file})
   
    ADD_CUSTOM_COMMAND(OUTPUT ${c_file}
        COMMAND lcm-gen -c ${CMAKE_CURRENT_SOURCE_DIR}/${lcm_file} --c-cpath ${CMAKE_CURRENT_SOURCE_DIR}/ --c-hpath ${CMAKE_CURRENT_SOURCE_DIR}/)
    SET(c_files ${c_files} ${c_file})
    
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${c_file} ${CMAKE_CURRENT_BINARY_DIR}/${c_file} COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${h_file} ${CMAKE_CURRENT_BINARY_DIR}/${h_file} COPYONLY)

endforeach(lcm_file)

set(LIBRARY_OUTPUT_PATH ../lib)

remove_definitions(-DDISABLE_LIBUSB-1.0)

add_library(so_lcm STATIC ${c_files}) 
