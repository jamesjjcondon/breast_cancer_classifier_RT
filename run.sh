#!/bin/bash

NUM_PROCESSES=16
DEVICE_TYPE='gpu'
NUM_EPOCHS=10
SHRINK_FACTOR=0
HEATMAP_BATCH_SIZE=1200 # AIML
#HEATMAP_BATCH_SIZE=500 # AHMS
GPU_NUMBER=-1

#BASE_DIR='/home/james/Documents/mydev/nyukat2.0' # AHMS
BASE_DIR='/home/mlim-user/Documents/james/my_dev/nyukat2.0' #AIML

CSVDIR='/home/mlim-user/Documents/james/tempdir/cleaning_temp' #AIML
#CSVDIR='/Data/james/bssa/temp' # AHMS

#DATA_FOLDER='/Data/james/NYU_val/images_5050' # AHMS
DATA_FOLDER='/data/james/NYU_retrain/test_ims' #NYU_mini_retrain' #AIML
#DATA_FOLDER='/media/james/drjc_ext_HD/data/AHMS_NYURT_test/test_ims' # AHMS

IMAGE_FOLDER=$DATA_FOLDER'/renamed_dicoms' 

echo "BASE_DIR, DATA_FOLDER, IMAGE_FOLDER:"
echo $BASE_DIR
echo $DATA_FOLDER
echo $IMAGE_FOLDER

INITIAL_EXAM_LIST_PATH=$DATA_FOLDER'/pre_crop_exam_list.pkl'
PATCH_MODEL_PATH=$BASE_DIR'/models/sample_patch_model.p'
IMAGE_MODEL_PATH=$BASE_DIR'/models/sample_image_model.p'
IMAGEHEATMAPS_MODEL_PATH=$BASE_DIR'/models/sample_imageheatmaps_model.p'

CROPPED_IMAGE_PATH=$DATA_FOLDER'/hm_cropped_ims_sf0'    
CROPPED_EXAM_LIST_PATH=$DATA_FOLDER'/cropped_exam_list_sf0.pkl'
EXAM_LIST_PATH=$DATA_FOLDER'/data_hm_sf0.pkl'
HEATMAPS_PATH=$DATA_FOLDER'/heatmaps_hm_sf0'
IMAGE_PREDICTIONS_PATH=$DATA_FOLDER'/5050_preds_hm_sf0/image_predictions.csv'
IMAGEHEATMAPS_PREDICTIONS_PATH=$DATA_FOLDER'/5050_preds_hm_sf0/imageheatmaps_predictions.csv'

#export PYTHONPATH=$(pwd):$PYTHONPATH

#echo 'Stage 1: Crop Mammograms'
#echo '(3600 ims took 18m with 22 processes on box)'
#time python3 src/cropping/crop_mammogram.py \
#    --input-data-folder $IMAGE_FOLDER \
#    --output-data-folder $CROPPED_IMAGE_PATH \
#    --exam-list-path $INITIAL_EXAM_LIST_PATH  \
#    --cropped-exam-list-path $CROPPED_EXAM_LIST_PATH  \
#    --num-processes $NUM_PROCESSES \
#    --re-training \
#    --df-fp $CSVDIR'/super.csv' \
    #--shrink-factor $SHRINK_FACTOR \

echo 'Stage 2: Extract Centers'
time python3 src/optimal_centers/get_optimal_centers.py \
    --cropped-exam-list-path $CROPPED_EXAM_LIST_PATH \
    --data-prefix $CROPPED_IMAGE_PATH \
    --output-exam-list-path $EXAM_LIST_PATH \
    --num-processes $NUM_PROCESSES
    --shrinking

echo 'Stage 3: Generate Heatmaps'
time python3 src/heatmaps/run_producer.py \
    --model-path $PATCH_MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --batch-size $HEATMAP_BATCH_SIZE \
    --output-heatmap-path $HEATMAPS_PATH \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER \
    --use-hdf5 
    #--shrink-factor $SHRINK_FACTOR


echo 'Stage 4a: Run Classifier (Image)'
time python3 src/modeling/run_model.py \
    --model-path $IMAGE_MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --output-path $IMAGE_PREDICTIONS_PATH \
    --use-augmentation \
    --num-epochs $NUM_EPOCHS \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER \
    --use-hdf5

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
    --use-hdf5
