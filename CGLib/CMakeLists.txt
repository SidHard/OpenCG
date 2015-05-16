################# OpenCG project files ################

SET(OpenCG_TOP_HEADERS
OpenCG.h
)

SET(OpenCG_CORE_SOURCES
Core/CGConvolution.cpp
Core/CGFilter.cpp
Core/CGPyramid.cpp
)

SET(OpenCG_CORE_HEADERS
Core/CGCore.h
Core/CGConfig.h
Core/CGImage.h
Core/CGConvolution.h
Core/CGFilter.h
Core/CGPyramid.h
)

set(OpenCG_CORE_CUDA_SOURCES
Core/CGConvolution.cu
Core/CGFilter.cu
Core/CGPyramid.cu
)

SET(OpenCG_HOG_HEADERS
Hog/CGHog.h
Hog/CGHogConfig.h
Hog/CGHogEngine.h
Hog/CGHogHistogram.h
Hog/CGHogSvmEvaluate.h
)

SET(OpenCG_HOG_SOURCES
Hog/CGHogEngine.cpp
Hog/CGHogHistogram.cpp
Hog/CGHogSvmEvaluate.cpp
)

set(OpenCG_HOG_CUDA_SOURCES
Hog/CGHogEngine.cu
Hog/CGHogHistogram.cu
Hog/CGHogSvmEvaluate.cu
)

###############       Collect Source      ###################

set(CGLIB_CUDA_CORE
	${OpenCG_CORE_SOURCES}
	${OpenCG_CORE_HEADERS}
	${OpenCG_CORE_CUDA_SOURCES}
)

set(CGLIB_CPU_CORE
	${OpenCG_CORE_SOURCES}
	${OpenCG_CORE_HEADERS}
)

set(CGLIB_CUDA_HOG
	${OpenCG_HOG_HEADERS}
	${OpenCG_HOG_SOURCES}
	${OpenCG_HOG_CUDA_SOURCES}
)

set(CGLIB_CPU_HOG
	${OpenCG_HOG_HEADERS}
	${OpenCG_HOG_SOURCES}
)


############### Specify the source groups ###################

SOURCE_GROUP("" FILES ${OpenCG_TOP_HEADERS})
SOURCE_GROUP(Core FILES ${OpenCG_CORE_HEADERS} ${OpenCG_CORE_CUDA_SOURCES} ${OpenCG_CORE_SOURCES})
SOURCE_GROUP(Hog FILES ${OpenCG_HOG_HEADERS} ${OpenCG_HOG_CUDA_SOURCES} ${OpenCG_HOG_SOURCES})

set(LIBRARY_OUTPUT_PATH, ${PROJECT_SOURCE_DIR}/lib)

IF(WITH_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  cuda_add_library(CGLib
	${OpenCG_TOP_HEADERS}
	${CGLIB_CUDA_CORE}
	${CGLIB_CUDA_HOG})
  target_link_libraries(CGLib ${CUDA_LIBRARY})

ELSE()
  add_definitions(-DCOMPILE_WITHOUT_CUDA)
  add_library(CGLib
	${OpenCG_TOP_HEADERS}
	${CGLIB_CPU_CORE}
	${CGLIB_CPU_HOG})
ENDIF()



