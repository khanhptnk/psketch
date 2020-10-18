#!/bin/bash

#DATA_DIR=$(realpath ../data)
CODE_DIR=$(realpath .)

#echo "Data dir: $DATA_DIR"
echo "Code dir: $CODE_DIR"

nvidia-docker run -it --volume $CODE_DIR:/root/mount/regex regex  

#nvidia-docker run -it --mount type=bind,source=$DATA_DIR,target=/root/mount/navlang/data,readonly \
#                      --volume $CODE_DIR:/root/mount/navlang/code \
#                      navlang /bin/bash 
