#!/bin/bash
cd src/pose

###--------------------------------------------------------------
DEVICES=0,
#DEVICES=0,1,2,3,


RUN_FILE='./tools/dist_test.sh'
PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))

# Accept dataset and model type as arguments
DATASET=${1:-"med_sapien"}
MODEL_TYPE=${2:-"lora_sapiens"} # Default to sapiens if not provided

## Map dataset to the correct checkpoint path
#declare -A CHECKPOINTS_SAPIENS
#CHECKPOINTS_SAPIENS["med_sapien"]="Outputs/train/med_sapien/sapiens_0.3b-210e_med_sapien-1024x768/node/12-09-2024_09:25:23/best_EPE_epoch_182.pth"


declare -A CHECKPOINTS_LORA_SAPIENS
CHECKPOINTS_LORA_SAPIENS["med_sapien"]="../../checkpoints/best_EPE_epoch_199.pth"

# Select the correct checkpoint mapping
if [ "$MODEL_TYPE" = "sapiens" ]; then
    CHECKPOINT=${CHECKPOINTS_SAPIENS[$DATASET]}
elif [ "$MODEL_TYPE" = "lora_sapiens" ]; then
    CHECKPOINT=${CHECKPOINTS_LORA_SAPIENS[$DATASET]}
else
    echo "Invalid model type: $MODEL_TYPE"
    exit 1
fi

if [ -z "$CHECKPOINT" ]; then
    echo "Invalid dataset: $DATASET"
    exit 1
fi

##---------MODEL_CARD----------------------------
MODEL="${MODEL_TYPE}_0.3b-210e_${DATASET}-1024x768"
JOB_NAME="test_pose_whole_$MODEL"
TEST_BATCH_SIZE_PER_GPU=4


###--------------------------------------------------------------
mode='multi-gpu'

###--------------------------------------------------------------+
CONFIG_FILE="configs/sapiens_pose/${DATASET}/${MODEL}.py"
OUTPUT_DIR="Outputs/test/${DATASET}/${MODEL}/"
DUMP_DIR="Outputs/test/${DATASET}/${MODEL}/test_dump.pkl"

export TF_CPP_MIN_LOG_LEVEL=2

## Set the options for the test
OPTIONS="$(echo "test_dataloader.batch_size=$TEST_BATCH_SIZE_PER_GPU")"

##--------------------------------------------------------------
## Run based on mode
if [ "$mode" = "debug" ]; then
    TEST_BATCH_SIZE_PER_GPU=16 ## Works for single GPU

    OPTIONS="$(echo "test_dataloader.batch_size=${TEST_BATCH_SIZE_PER_GPU} test_dataloader.num_workers=0 test_dataloader.persistent_workers=False")"
    CUDA_VISIBLE_DEVICES=${DEVICES} python tools/test.py ${CONFIG_FILE} ${CHECKPOINT} --work-dir ${OUTPUT_DIR} --cfg-options ${OPTIONS}

elif [ "$mode" = "multi-gpu" ]; then
    NUM_GPUS_STRING_LEN=${#DEVICES}
    NUM_GPUS=$((NUM_GPUS_STRING_LEN/2))

    LOG_FILE="$(echo "${OUTPUT_DIR}/log.txt")"
    mkdir -p ${OUTPUT_DIR}; touch ${LOG_FILE}

    CUDA_VISIBLE_DEVICES=${DEVICES} PORT=${PORT} ${RUN_FILE} ${CONFIG_FILE} ${CHECKPOINT} \
            ${NUM_GPUS} \
            --work-dir ${OUTPUT_DIR} \
            --cfg-options ${OPTIONS} \
            --dump ${DUMP_DIR} \
            | tee ${LOG_FILE}
fi
