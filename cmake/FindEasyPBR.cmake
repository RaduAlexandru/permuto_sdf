if(EASYPBR_FOUND)
    return()
endif()


#get the path of site packages which is where easypbr got installed https://stackoverflow.com/a/31384782 and https://stackoverflow.com/a/40006251
execute_process(
  COMMAND "python3" -c "if True:
    import sys
    print( next(p for p in sys.path if 'site-packages' in p))"
  OUTPUT_VARIABLE PYTHON_SITE
  OUTPUT_STRIP_TRAILING_WHITESPACE)
# message("--------------------PYTHON is at ", ${PYTHON_SITE})
#Get only the first line of the egg_link because that is pointing towards the easypbrsrc
execute_process (
    # COMMAND bash -c "date +'%F %T'" "${PYTHON_SITE}/easypbr.egg-link"
    COMMAND bash -c "head -1 $0" "${PYTHON_SITE}/easypbr.egg-link"
    OUTPUT_VARIABLE EASYPBR_SRC_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# message("EASYPPBR source is ", ${EASYPBR_SRC_PATH})

#DEBUG
# set(EASYPBR_SRC_PATH "/media/rosu/Data/phd/c_ws/src/easy_pbr")

find_path(EASYPBR_CORE_INCLUDE_DIR  NAMES easy_pbr/Viewer.h PATHS "${EASYPBR_SRC_PATH}/include" )
find_path(EASYPBR_EASYGL_INCLUDE_DIR  NAMES easy_gl/Texture2D.h PATHS "${EASYPBR_SRC_PATH}/deps/easy_gl/include" )
find_path(EASYPBR_LOGURU_INCLUDE_DIR  NAMES loguru.hpp PATHS "${EASYPBR_SRC_PATH}/deps/loguru" )
find_path(EASYPBR_CONFIGURU_INCLUDE_DIR  NAMES configuru.hpp PATHS "${EASYPBR_SRC_PATH}/deps/configuru" )
find_path(EASYPBR_UTILS_INCLUDE_DIR  NAMES Profiler.h PATHS "${EASYPBR_SRC_PATH}/deps/utils/include" )
find_path(EASYPBR_CONCURRENTQUEUE_INCLUDE_DIR  NAMES concurrentqueue.h PATHS "${EASYPBR_SRC_PATH}/deps/concurrent_queue" )
find_path(EASYPBR_IMGUI_INCLUDE_DIR  NAMES imgui.h PATHS "${EASYPBR_SRC_PATH}/deps/imgui" )
find_path(EASYPBR_IMGUIZMO_INCLUDE_DIR  NAMES ImGuizmo.h PATHS "${EASYPBR_SRC_PATH}/deps/imguizmo" )
find_path(EASYPBR_BETTERENUM_INCLUDE_DIR  NAMES enum.h PATHS "${EASYPBR_SRC_PATH}/deps/better_enums" )
find_path(EASYPBR_EASYPYTORCH_INCLUDE_DIR  NAMES UtilsPytorch.h PATHS "${EASYPBR_SRC_PATH}/deps/easy_pytorch" )
find_path(EASYPBR_PYBIND_INCLUDE_DIR  NAMES pybind11/pybind11.h PATHS "${EASYPBR_SRC_PATH}/deps/pybind11/include" ) #important because any module that tries to have python modules that interact with easypbr should be built with the same pybind version
find_path(EASYPBR_TINYPLY_INCLUDE_DIR  NAMES tinyply.h PATHS "${EASYPBR_SRC_PATH}/deps/tiny_ply/source" )

set( EASYPBR_INCLUDE_DIR ${EASYPBR_SRC_PATH}/extern ${EASYPBR_CORE_INCLUDE_DIR} ${EASYPBR_EASYGL_INCLUDE_DIR} ${EASYPBR_LOGURU_INCLUDE_DIR}
                                                    ${EASYPBR_CONFIGURU_INCLUDE_DIR} ${EASYPBR_UTILS_INCLUDE_DIR} ${EASYPBR_CONCURRENTQUEUE_INCLUDE_DIR}
                                                    ${EASYPBR_IMGUI_INCLUDE_DIR} ${EASYPBR_IMGUIZMO_INCLUDE_DIR} ${EASYPBR_BETTERENUM_INCLUDE_DIR}
                                                    ${EASYPBR_EASYPYTORCH_INCLUDE_DIR}  ${EASYPBR_PYBIND_INCLUDE_DIR}  ${EASYPBR_TINYPLY_INCLUDE_DIR}
                                                    )

# message("--------------------EASYPPBR include dir is at ", ${EASYPBR_INCLUDE_DIR})


find_library(EASYPBR_LIBRARY
    NAMES libeasypbr_cpp.so
    HINTS ${EASYPBR_SRC_PATH}
    DOC "The EasyPBR lib directory"
    NO_DEFAULT_PATH)
# message("--------------------EASYPPBR lib dir is at ", ${EASYPBR_LIBRARY})

add_definitions( -DDEFAULT_CONFIG="${EASYPBR_SRC_PATH}/config/default_params.cfg" )
