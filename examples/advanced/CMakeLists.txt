file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cu )
foreach( testsourcefile ${SOURCES} )
    # I used a simple string replace, to cut off .cpp.
    string( REPLACE ".cu" "" testname ${testsourcefile} )
    add_executable( ${testname} ${testsourcefile} )
    set_target_properties(${testname} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endforeach( testsourcefile ${SOURCES} )
