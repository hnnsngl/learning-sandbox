add_executable( mnist-static mnist-static.cpp loadweights.cpp )
target_link_libraries( mnist-static ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})
install (TARGETS mnist-static DESTINATION bin)

add_executable( mnist-flex mnist-flex.cpp loadweights.cpp options.cpp )
target_link_libraries( mnist-flex ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})
install (TARGETS mnist-flex DESTINATION bin)
