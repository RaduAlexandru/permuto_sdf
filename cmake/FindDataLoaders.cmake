if(EASYPBR_FOUND)
    return()
endif()


#get the path of site packages which is where easypbr got installed https://stackoverflow.com/a/31384782 and https://stackoverflow.com/a/40006251
execute_process(
  COMMAND "python" -c "if True:
    import sys
    print( next(p for p in sys.path if 'site-packages' in p))"
  OUTPUT_VARIABLE PYTHON_SITE
  OUTPUT_STRIP_TRAILING_WHITESPACE)
# message("--------------------PYTHON is at " ${PYTHON_SITE})
#Get only the first line of the egg_link because that is pointing towards the easypbrsrc
execute_process (
    # COMMAND bash -c "date +'%F %T'" "${PYTHON_SITE}/easypbr.egg-link"
    COMMAND bash -c "head -1 $0" "${PYTHON_SITE}/dataloaders.egg-link"
    OUTPUT_VARIABLE DATALOADERS_SRC_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# message("DATALOADERS source is " ${DATALOADERS_SRC_PATH})

#DEBUG
# set(EASYPBR_SRC_PATH "/media/rosu/Data/phd/c_ws/src/easy_pbr")

find_path(DATALOADERS_CORE_INCLUDE_DIR  NAMES data_loaders/DataLoaderShapeNetPartSeg.h PATHS "${DATALOADERS_SRC_PATH}/include" )
# find_path(EASYPBR_EASYGL_INCLUDE_DIR  NAMES Texture2D.h PATHS "${EASYPBR_SRC_PATH}/deps/easy_gl" )
# find_path(EASYPBR_LOGURU_INCLUDE_DIR  NAMES loguru.hpp PATHS "${EASYPBR_SRC_PATH}/deps/loguru" )
# find_path(EASYPBR_CONFIGURU_INCLUDE_DIR  NAMES configuru.hpp PATHS "${EASYPBR_SRC_PATH}/deps/configuru" )
# find_path(EASYPBR_UTILS_INCLUDE_DIR  NAMES Profiler.h PATHS "${EASYPBR_SRC_PATH}/deps/utils/include" )
# find_path(EASYPBR_CONCURRENTQUEUE_INCLUDE_DIR  NAMES concurrentqueue.h PATHS "${EASYPBR_SRC_PATH}/deps/concurrent_queue" )
# find_path(EASYPBR_IMGUI_INCLUDE_DIR  NAMES imgui.h PATHS "${EASYPBR_SRC_PATH}/deps/imgui" )
# find_path(EASYPBR_IMGUIZMO_INCLUDE_DIR  NAMES ImGuizmo.h PATHS "${EASYPBR_SRC_PATH}/deps/imguizmo" )
# find_path(EASYPBR_BETTERENUM_INCLUDE_DIR  NAMES enum.h PATHS "${EASYPBR_SRC_PATH}/deps/better_enums" )

# set( EASYPBR_INCLUDE_DIR ${EASYPBR_SRC_PATH}/extern ${EASYPBR_CORE_INCLUDE_DIR} ${EASYPBR_EASYGL_INCLUDE_DIR} ${EASYPBR_LOGURU_INCLUDE_DIR} ${EASYPBR_CONFIGURU_INCLUDE_DIR} ${EASYPBR_UTILS_INCLUDE_DIR} ${EASYPBR_CONCURRENTQUEUE_INCLUDE_DIR} ${EASYPBR_IMGUI_INCLUDE_DIR} ${EASYPBR_IMGUIZMO_INCLUDE_DIR} ${EASYPBR_BETTERENUM_INCLUDE_DIR} )
set( DATALOADERS_INCLUDE_DIR ${DATALOADERS_CORE_INCLUDE_DIR}  )

# message("--------------------EASYPPBR include dir is at ", ${EASYPBR_INCLUDE_DIR})


find_library(DATALOADERS_LIBRARY
    NAMES libdataloaders_cpp.so
    HINTS ${DATALOADERS_SRC_PATH}
    DOC "The DataLoaders lib directory"
    NO_DEFAULT_PATH)
# message("--------------------DATALOADERS lib dir is at " ${DATALOADERS_LIBRARY})


