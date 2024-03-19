#!/bin/bash

# this is for one of 'train', 'val' or 'test'
NUM_PROCESSES=12
DEVICE_TYPE='gpu'
SHRINK_FACTOR='match'
HEATMAP_BATCH_SIZE=800 # AIML
#HEATMAP_BATCH_SIZE=500 # AHMS
GPU_NUMBER=-1

#BASE_DIR='/home/james/Documents/mydev/nyukat2.0' # AHMS
BASE_DIR='/home/mlim-user/Documents/james/my_dev/nyukat2.0' #AIML
CSVDIR='/home/mlim-user/Documents/james/tempdir/cleaning_temp' #AIML
#CSVDIR='/Data/james/bssa/temp' # AHMS

DATA_FOLDER='/data/james/NYU_retrain/test_ims_master/size_matched'
#DATA_FOLDER='/Data/james/NYU_val/images_5050' # AHMS
#DATA_FOLDER='/data/james/NYU_retrain/test_ims_master/NY_16-8pc_incidence' #AIML HDD
#DATA_FOLDER='/nvme/james/test_ims_master/NY_16-8pc_incidence' #AIML nvme
#DATA_FOLDER='/media/james/drjc_ext_HD/data/AHMS_NYURT_test/test_ims' # AHMS

IMAGE_FOLDER='/data/james/NYU_retrain/test_ims_master/renamed_dicoms' 

echo "BASE_DIR, DATA_FOLDER, IMAGE_FOLDER:"
echo $BASE_DIR
echo $DATA_FOLDER
echo $IMAGE_FOLDER

INITIAL_EXAM_LIST_PATH='/data/james/NYU_retrain/test_ims_master/pre_crop_exam_list.pkl'
PATCH_MODEL_PATH=$BASE_DIR'/models/sample_patch_model.p'
IMAGE_MODEL_PATH=$BASE_DIR'/models/sample_image_model.p'
IMAGEHEATMAPS_MODEL_PATH=$BASE_DIR'/models/sample_imageheatmaps_model.p'

CROPPED_IMAGE_PATH=$DATA_FOLDER'/cropped_images_sf_matched'    
CROPPED_EXAM_LIST_PATH=$DATA_FOLDER'/cropped_exam_list_sf_matched.pkl'
EXAM_LIST_PATH=$DATA_FOLDER'/data_sf_matched_NYU_centre.pkl'
HEATMAPS_PATH=$DATA_FOLDER'/heatmaps_sf_matched'

export PYTHONPATH=$(pwd):$PYTHONPATH

#echo 'Stage 1: Crop Mammograms'
#echo '(3600 ims took 18m with 22 processes on box)'
#time python3 src/cropping/crop_mammogram.py \
#    --input-data-folder $IMAGE_FOLDER \
#    --output-data-folder $CROPPED_IMAGE_PATH \
#    --exam-list-path $INITIAL_EXAM_LIST_PATH  \
#    --cropped-exam-list-path $CROPPED_EXAM_LIST_PATH  \
#    --num-processes $NUM_PROCESSES \
#    --shrink-factor $SHRINK_FACTOR \
#    --re-training \
#    --df-fp $CSVDIR'/super.csv'
    
#echo 'Stage 2: Extract Centers'
#time python3 src/optimal_centers/get_optimal_centers.py \
#    --cropped-exam-list-path $CROPPED_EXAM_LIST_PATH \
#    --data-prefix $CROPPED_IMAGE_PATH \
#    --output-exam-list-path $EXAM_LIST_PATH \
#    --num-processes $NUM_PROCESSES \
#    --NYU-centre

echo 'Stage 3: Generate Heatmaps'
time python3 src/heatmaps/run_producer.py \
    --model-path $PATCH_MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --batch-size $HEATMAP_BATCH_SIZE \
    --output-heatmap-path $HEATMAPS_PATH \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER \
    --use-hdf5 \
    --shrink-factor $SHRINK_FACTOR
