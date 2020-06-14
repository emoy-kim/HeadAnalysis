target_link_libraries(HeadAnalysis glad glfw3dll)

if(${CMAKE_BUILD_TYPE} MATCHES Debug)
	target_link_libraries(
		HeadAnalysis 
			dlib19.20.0_debug_64bit_msvc1926
			opencv_cored 
			opencv_imgprocd 
			opencv_imgcodecsd 
			opencv_highguid
			opencv_calib3dd
	)
else()
	target_link_libraries(
		HeadAnalysis 
			dlib19.20.0_release_64bit_msvc1926
			opencv_core 
			opencv_imgproc 
			opencv_imgcodecs 
			opencv_highgui
			opencv_calib3d
	)
endif()