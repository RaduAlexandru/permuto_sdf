cmake_minimum_required(VERSION 3.18) 

project(
    permuto_sdf
    VERSION 1.0
    LANGUAGES CXX CUDA
)


###   VARIABLES   ##############################################################
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall -O3") 
set(CMAKE_CXX_STANDARD 17) #we need c++17 because this solves alignment issues with eigen http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1409
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)








####   GLOBAL OPTIONS   ###https://stackoverflow.com/questions/15201064/cmake-conditional-preprocessor-define-on-code
set(PSDF_CUDA_ARCHITECTURES "" CACHE STRING "Build PermutoSDF for a specific GPU architecture.")
option(PSDF_WITH_GL "Compile with GL" OFF) #sometimes when we run headless we would like to disable all the opengl stuff because we are not creating the Viewer and therefore do not initialize a OpenGL context and therefore the PermutoSDF class cannot create opengl textures or anything like that 



######   PACKAGES   ############################################################
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs calib3d )
find_package(EasyPBR REQUIRED)
find_package(DataLoaders REQUIRED) #needed for the TensorReel
find_package(CUDA REQUIRED)
add_subdirectory(${EASYPBR_SRC_PATH}/deps/pybind11  [EXCLUDE_FROM_ALL])
# get and append paths for finding dep
execute_process( #do it like this https://github.com/facebookresearch/hanabi_SAD/blob/6e4ed590f5912fcb99633f4c224778a3ba78879b/rela/CMakeLists.txt#L10
  COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE TorchPath
)
if(TorchPath STREQUAL "")
    set(TORCH_FOUND False)
else()
    list(APPEND CMAKE_PREFIX_PATH ${TorchPath})
    find_package(Torch)
	#check that the CXX_ABI matches because otherwise we will have a ton of linking errors https://github.com/pytorch/pytorch/issues/51039
    execute_process( 
        COMMAND python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
        OUTPUT_VARIABLE TORCH_USES_CXX11_ABI
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    message("TORCH_USES_CXX11_ABI: " ${TORCH_USES_CXX11_ABI})
    if(TORCH_USES_CXX11_ABI STREQUAL "True")
        #all good
    else()
        message( FATAL_ERROR "Torch was compiled with CXX11_ABI=0. This means it cannot be linked with the other libraries that EasyPBR uses. The best would be to compile PyTorch from source. For more detail see https://github.com/pytorch/pytorch/issues/51039" )
    endif()
endif()



###   INCLUDES   #########################################################
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/kernels)
include_directories(${CMAKE_SOURCE_DIR}/deps)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${EASYPBR_INCLUDE_DIR})
include_directories(${DATALOADERS_INCLUDE_DIR}) #needed for the TensorReel
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})


#pybind
pybind11_add_module(permuto_sdf ${PROJECT_SOURCE_DIR}/src/PyBridge.cxx )




###############################################################################
# CUDA compiler setup
###############################################################################

set(CUDA_NVCC_FLAGS "") #start with a clean state and don't inherit things from pytorch

# Figure out CUDA version
if(CMAKE_CUDA_COMPILER_LOADED)
	if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION MATCHES "^([0-9]+\\.[0-9]+)")
		set(CUDA_VERSION "${CMAKE_MATCH_1}")
	endif()
endif()

# Adapted from the CMake source code at https://github.com/Kitware/CMake/blob/master/Modules/FindCUDA/select_compute_arch.cmake
# Simplified to return a semicolon-separated list of the compute capabilities of installed devices
function(PSDF_AUTODETECT_CUDA_ARCHITECTURES OUT_VARIABLE)
	if (NOT PSDF_AUTODETECT_CUDA_ARCHITECTURES_OUTPUT)
		if (CMAKE_CUDA_COMPILER_LOADED) # CUDA as a language
			set(file "${PROJECT_BINARY_DIR}/detect_PSDF_cuda_architectures.cu")
		else()
			set(file "${PROJECT_BINARY_DIR}/detect_PSDF_cuda_architectures.cpp")
		endif()

		file(WRITE ${file} ""
			"#include <cuda_runtime.h>\n"
			"#include <cstdio>\n"
			"int main() {\n"
			"	int count = 0;\n"
			"	if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
			"	if (count == 0) return -1;\n"
			"	for (int device = 0; device < count; ++device) {\n"
			"		cudaDeviceProp prop;\n"
			"		if (cudaSuccess == cudaGetDeviceProperties(&prop, device)) {\n"
			"			std::printf(\"%d%d\", prop.major, prop.minor);\n"
			"			if (device < count - 1) std::printf(\";\");\n"
			"		}\n"
			"	}\n"
			"	return 0;\n"
			"}\n"
		)

		if (CMAKE_CUDA_COMPILER_LOADED) # CUDA as a language
			try_run(run_result compile_result ${PROJECT_BINARY_DIR} ${file} RUN_OUTPUT_VARIABLE compute_capabilities)
		else()
			try_run(
				run_result compile_result ${PROJECT_BINARY_DIR} ${file}
				CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CUDA_INCLUDE_DIRS}"
				LINK_LIBRARIES ${CUDA_LIBRARIES}
				RUN_OUTPUT_VARIABLE compute_capabilities
			)
		endif()

		if (run_result EQUAL 0)
			# If the user has multiple GPUs with the same compute capability installed, list that capability only once.
			list(REMOVE_DUPLICATES compute_capabilities)
			set(PSDF_AUTODETECT_CUDA_ARCHITECTURES_OUTPUT ${compute_capabilities} CACHE INTERNAL "Returned GPU architectures from detect_gpus tool" FORCE)
		endif()
	endif()

	if (NOT PSDF_AUTODETECT_CUDA_ARCHITECTURES_OUTPUT)
		message(STATUS "Automatic GPU detection failed. Building for Turing and Ampere as a best guess.")
		set(${OUT_VARIABLE} "75;86" PARENT_SCOPE)
	else()
		set(${OUT_VARIABLE} ${PSDF_AUTODETECT_CUDA_ARCHITECTURES_OUTPUT} PARENT_SCOPE)
	endif()
endfunction()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

get_directory_property(PSDF_HAS_PARENT PARENT_DIRECTORY)
if (DEFINED ENV{PSDF_CUDA_ARCHITECTURES})
	message(STATUS "Obtained CUDA architectures from environment variable PSDF_CUDA_ARCHITECTURES=$ENV{PSDF_CUDA_ARCHITECTURES}")
	set(CMAKE_CUDA_ARCHITECTURES $ENV{PSDF_CUDA_ARCHITECTURES})
elseif (PSDF_CUDA_ARCHITECTURES)
	message(STATUS "Obtained CUDA architectures from CMake variable PSDF_CUDA_ARCHITECTURES=${PSDF_CUDA_ARCHITECTURES}")
	set(CMAKE_CUDA_ARCHITECTURES ${PSDF_CUDA_ARCHITECTURES})
else()
	message(STATUS "Obtained CUDA architectures automatically from installed GPUs")
	PSDF_AUTODETECT_CUDA_ARCHITECTURES(CMAKE_CUDA_ARCHITECTURES)
endif()

# If the CUDA version does not support the chosen architecture, target
# the latest supported one instead.
if (CUDA_VERSION VERSION_LESS 11.0)
	set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 75)
elseif (CUDA_VERSION VERSION_LESS 11.1)
	set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 80)
elseif (CUDA_VERSION VERSION_LESS 11.8)
	set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 86)
else()
	set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 90)
endif()

if (CUDA_VERSION VERSION_GREATER_EQUAL 12.0)
	set(EARLIEST_SUPPORTED_CUDA_ARCHITECTURE 50)
else()
	set(EARLIEST_SUPPORTED_CUDA_ARCHITECTURE 20)
endif()

foreach (CUDA_CC IN LISTS CMAKE_CUDA_ARCHITECTURES)
	if (CUDA_CC GREATER ${LATEST_SUPPORTED_CUDA_ARCHITECTURE})
		message(WARNING "CUDA version ${CUDA_VERSION} is too low for detected architecture ${CUDA_CC}. Targeting the highest supported architecture ${LATEST_SUPPORTED_CUDA_ARCHITECTURE} instead.")
		list(REMOVE_ITEM CMAKE_CUDA_ARCHITECTURES ${CUDA_CC})
		if (NOT CMAKE_CUDA_ARCHITECTURES)
			list(APPEND CMAKE_CUDA_ARCHITECTURES ${LATEST_SUPPORTED_CUDA_ARCHITECTURE})
		endif()
	endif()

	if (CUDA_CC LESS ${EARLIEST_SUPPORTED_CUDA_ARCHITECTURE})
		message(ERROR "CUDA version ${CUDA_VERSION} no longer supports detected architecture ${CUDA_CC}. Targeting the lowest supported architecture ${EARLIEST_SUPPORTED_CUDA_ARCHITECTURE} instead.")
		list(REMOVE_ITEM CMAKE_CUDA_ARCHITECTURES ${CUDA_CC})
		if (NOT CMAKE_CUDA_ARCHITECTURES)
			list(APPEND CMAKE_CUDA_ARCHITECTURES ${EARLIEST_SUPPORTED_CUDA_ARCHITECTURE})
		endif()
	endif()
endforeach(CUDA_CC)

if (NOT CMAKE_CUDA_ARCHITECTURES)
	list(APPEND CMAKE_CUDA_ARCHITECTURES ${LATEST_SUPPORTED_CUDA_ARCHITECTURE})
endif()

# Sort the list to obtain lowest architecture that must be compiled for.
list(SORT CMAKE_CUDA_ARCHITECTURES COMPARE NATURAL ORDER ASCENDING)
list(GET CMAKE_CUDA_ARCHITECTURES 0 MIN_GPU_ARCH)

string(REPLACE "-virtual" "" MIN_GPU_ARCH "${MIN_GPU_ARCH}")

message(STATUS "Targeting CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
if (PSDF_HAS_PARENT)
	set(PSDF_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} PARENT_SCOPE)
	set(PSDF_CUDA_VERSION ${CUDA_VERSION} PARENT_SCOPE)
endif()


if (CUDA_VERSION VERSION_LESS 10.2)
	message(FATAL_ERROR "CUDA version too low. permuto_sdf require CUDA 10.2 or higher.")
endif()

list(APPEND PSDF_LIBRARIES cuda)
list(APPEND PSDF_DEFINITIONS -DPSDF_MIN_GPU_ARCH=${MIN_GPU_ARCH})
if (PSDF_ALLOW_CUBLAS_CUSOLVER AND CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
	# Only compile the shampoo optimizer if
	# a new enough cuBLAS version is available.
	list(APPEND PSDF_LIBRARIES cublas)
endif()

if (PSDF_HAS_PARENT)
	set(PSDF_DEFINITIONS ${PSDF_DEFINITIONS} PARENT_SCOPE)
endif()

if (MSVC)
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=/bigobj")
else()
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
	list(APPEND CUDA_NVCC_FLAGS "-Xcudafe=--diag_suppress=unrecognized_gcc_pragma")
endif()
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
# list(APPEND CUDA_NVCC_FLAGS "-Xcudafe=--diag_suppress=20236")
list(APPEND CUDA_NVCC_FLAGS "--disable-warnings") #eigen throws too many warnings and the previous line with --diag_suppress=20236 doesn't work on all cuda versions







###############################################################################
# PermutoSDF library, samples, and benchmarks
###############################################################################

set(PSDF_SOURCES
    ${CMAKE_SOURCE_DIR}/src/PermutoSDF.cu  
    ${CMAKE_SOURCE_DIR}/src/Sphere.cu  
    # ${CMAKE_SOURCE_DIR}/src/VoxelGrid.cu  
    ${CMAKE_SOURCE_DIR}/src/OccupancyGrid.cu  
    ${CMAKE_SOURCE_DIR}/src/RaySampler.cu  
    ${CMAKE_SOURCE_DIR}/src/VolumeRendering.cu  
    ${CMAKE_SOURCE_DIR}/src/RaySamplesPacked.cu  
    ${CMAKE_SOURCE_DIR}/src/NGPGui.cxx
    ${CMAKE_SOURCE_DIR}/src/TrainParams.cxx
)


###############################################################################
# Linker / library
###############################################################################
# message("CMAKE_CXX_FLAGS--------------------------------------"${CMAKE_CXX_FLAGS})
# message("CUDA_NVCC_FLAGS--------------------------------------"${CUDA_NVCC_FLAGS})
message(STATUS "Targeting GPU architectures CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

add_library(permuto_sdf_cu SHARED ${PSDF_SOURCES})
# target_compile_definitions(permuto_sdf_cu PUBLIC ${PSDF_DEFINITIONS})
target_compile_options(permuto_sdf_cu PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)












###   LIBS   ###############################################
if(${TORCH_FOUND})
    set(LIBS ${LIBS} ${TORCH_LIBRARIES} )
    #torch 1.5.0 and above mess with pybind and we therefore need to link against libtorch_python.so also
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
    message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
    if(TORCH_PYTHON_LIBRARY)
        message(STATUS "Linking to torch_python_library")
        set(LIBS ${LIBS} ${TORCH_PYTHON_LIBRARY} )
    endif()
endif()
# set(LIBS ${LIBS} ${catkin_LIBRARIES} ${EASYPBR_LIBRARY} ${DATALOADERS_LIBRARY} ${CUDA_LIBRARIES} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set(LIBS ${LIBS} ${catkin_LIBRARIES} ${EASYPBR_LIBRARY} ${CUDA_LIBRARIES} ${TORCH_LIBRARIES} ${OpenCV_LIBS})


#####SET all global options
if(${PSDF_WITH_GL})
    message("USING GL")
    target_compile_definitions(permuto_sdf_cu PUBLIC PSDF_WITH_GL)
else()
    message("NOT USING GL")
endif()


target_link_libraries(permuto_sdf_cu ${LIBS} )
target_link_libraries(permuto_sdf PRIVATE permuto_sdf_cu )


#definitions for cmake variables that are necesarry during runtime
target_compile_definitions(permuto_sdf_cu PRIVATE PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}") #point to the cmakelist folder of the easy_pbr
target_compile_definitions(permuto_sdf_cu PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}") # points to the CMakeList folder of whichever project included easy_pbr


