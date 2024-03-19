#!/bin/bash

NUM_PROCESSES=6
DEVICE_TYPE='gpu'
NUM_EPOCHS=10
GPU_NUMBER=0

BASE_FOLDER='/media/james/drjc_ext_HD1/data/NYU_retrain' #HOME
#BASE_FOLDER='/Data/james/NYU_retrain' #AHMS
#BASE_FOLDER='/media/drjc/drjc_ext_HD/data/NYU_retrain' # yoga

#DATA_FOLDER='/Data/james/NYU_val/images_5050' # AHMS
#DATA_FOLDER='/data/james/NYU_exval/images_5050' #AIML
#DATA_FOLDER=$BASE_FOLDER'/val_ims_mini' #HOME
DATA_FOLDER=$BASE_FOLDER'/val_ims_sf2'

IMAGE_FOLDER=$DATA_FOLDER'/cropped_images_sf2' 

echo $DATA_FOLDER
echo $IMAGE_FOLDER

IMAGEHEATMAPS_MODEL_PATH=$BASE_FOLDER'/train_models/ckpts_19Mar_v1/19Mar_v2_epo=33_tloss=1.420_avloss=2.186.ckpt'

CROPPED_IMAGE_PATH=$IMAGE_FOLDER
CROPPED_EXAM_LIST_PATH=$BASE_FOLDER'/cropped_exam_list_sf2.pkl'
EXAM_LIST_PATH=$DATA_FOLDER'/data_sf2.pkl'
HEATMAPS_PATH=$DATA_FOLDER'/heatmaps_sf2'
IMAGE_PREDICTIONS_PATH=$BASE_FOLDER'/test_on_val/5050_preds/image_predictions.csv'
IMAGEHEATMAPS_PREDICTIONS_PATH=$BASE_FOLDER'/test_on_val/5050_preds/imageheatmaps_predictions.csv'

#export PYTHONPATH=$(pwd):$PYTHONPATH

echo 'Stage 4a: Run Classifier (Image)'
time python3 src/modeling/run_model.py \
    --model-path $IMAGEHEATMAPS_MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --output-path $IMAGE_PREDICTIONS_PATH \
    --use-augmentation \
    --num-epochs $NUM_EPOCHS \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER \
    --use-hdf5 \
    --retrained \
    --check-n-exams 20

echo 'Stage 4b: Run Classifier (Image+Heatmaps)'
time python3 src/modeling/run_model.py \
    --model-path $IMAGEHEATMAPS_MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --output-path $IMAGEHEATMAPS_PREDICTIONS_PATH \
    --use-heatmaps \
    --heatmaps-path $HEATMAPS_PATH \
    --use-augmentation \
    --num-epochs $NUM_EPOCHS \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER \
    --use-hdf5 \
    --retrained \
    --check-n-exams 20
