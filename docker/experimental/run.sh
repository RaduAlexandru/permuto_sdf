#!/usr/bin/env bash

# Check args
if [ "$#" -ne 0 ]; then
	  echo "usage: ./run.sh"
	    return 1
    fi

    # Get this script's path
    pushd `dirname $0` > /dev/null
    SCRIPTPATH=`pwd`
    popd > /dev/null

    set -e

	# for more info see: https://medium.com/@benjamin.botto/opengl-and-cuda-applications-in-docker-af0eece000f1
	# for more info see: https://gist.github.com/RafaelPalomar/f594933bb5c07184408c480184c2afb4
    # Run the container with shared X11
    docker run\
		--rm \
		--shm-size 12G\
		--gpus all\
		--net host\
		-e SHELL\
		-e DISPLAY=$DISPLAY\
		-e DOCKER=1\
		--volume="/etc/group:/etc/group:ro" \
		--volume="/etc/passwd:/etc/passwd:ro" \
		--volume="/etc/shadow:/etc/shadow:ro" \
		--volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
		--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
		--volume="/media/rosu/Data:/media/rosu/Data:rw"\
		--name permuto_sdf_img\
		-it permuto_sdf_img:latest


		# --volume="$HOME:$HOME:rw"\