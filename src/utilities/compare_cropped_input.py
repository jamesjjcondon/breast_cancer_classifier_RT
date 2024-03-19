#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 12:31:18 2020

Visually compares NYU v BSSA model input

@author: james
"""
import os
import random
import matplotlib.pyplot as plt
from src.utilities import pickling, reading_images
from src.utilities.all_utils import show_2 #, show
from src.data_loading import loading
from src.constants import DATADIR, VIEWS, REPODIR

#DATADIR = '/nvme/james'
#%%

#NYU_im_fp = os.path.join(REPODIR, 'sample_data/images')
#NYU_im_fp = '/home/james/Documents/mydev/breast_cancer_classifier/sample_output/cropped_images'
#NYU_im_fp = '/home/mlim-user/Documents/james/my_dev/breast_cancer_classifier/sample_output/cropped_images'

NYU_im_fp = '/home/james/mydev/breast_cancer_classifier/sample_output/cropped_images'
NYU_ims = sorted(os.listdir(NYU_im_fp))
#NYU_exams = pickling.unpickle_from_file('/home/james/Documents/mydev/breast_cancer_classifier/sample_data/exam_list_before_cropping.pkl')
NYU_exams = pickling.unpickle_from_file('/home/james/mydev/breast_cancer_classifier/sample_data/exam_list_before_cropping.pkl')


""" randomly matched to NYU pixel_array sizes """
#BSSA_im_fp = '/data/james/NYU_retrain/test_ims_master/test_ims_mini/size_matched/cropped_images_sf_matched'
#os.path.join(NVMEDIR, 'test_ims_master/cropped_images_sf2') #DATADIR, 'test_ims_master/renamed_dicoms')

""" matched to largest NYU pixel array height """
BSSA_im_fp = '/data/james/NYU_retrain/test_ims_master/large_matched/cropped_images_sf_matched_large'

""" matched to medium NYU pixel array height """
BSSA_im_fp = '/data/james/NYU_retrain/test_ims_master/small_matched/cropped_ims_NYU_small'

""" Downscaled to model window """
BSSA_im_fp = '/data/james/NYU_Retrain/test_ims_master/window_matched/cropped_images_sf_window'

""" Downscaled to smalles NYU pixel array height """
BSSA_im_fp = '/data/james/NYU_Retrain/test_ims_master/test_ims_mini/size_matched/cropped_images_sf_matched'

""" Downscaled by 2 / half """
#BSSA_im_fp = '/data/james/NYU_retrain/test_ims_master/cropped_images_sf2'


BSSA_ims = sorted(os.listdir(BSSA_im_fp))

all_BSSA_exams = pickling.unpickle_from_file(
        os.path.join(DATADIR, 'test_ims_master/test_ims_mini/pre_crop_exam_list.pkl')
        )


BSSA_exams = random.sample(all_BSSA_exams, 4)

def get_files_by_file_size(dirname, reverse=False):
    """ Return list of file paths in directory sorted by file size """

    # Get list of files
    filepaths = []
    for basename in os.listdir(dirname):
        filename = os.path.join(dirname, basename)
        if os.path.isfile(filename):
            filepaths.append(filename)

    # Re-populate list with filename, size tuples
    for i in range(len(filepaths)):
        filepaths[i] = (filepaths[i], os.path.getsize(filepaths[i]))

    # Sort list by file size
    # If reverse=True sort from largest to smallest
    # If reverse=False sort from smallest to largest
    filepaths.sort(key=lambda filename: filename[1], reverse=reverse)

    # Re-populate list with just filenames
    for i in range(len(filepaths)):
        filepaths[i] = filepaths[i][0]

    return filepaths

NYU_ims = get_files_by_file_size(NYU_im_fp, reverse=True)
BSSA_ims = get_files_by_file_size(BSSA_im_fp, reverse=True)#[:len(NYU_ims)]

#NYU_im_fp = '/home/james/Documents/mydev/breast_cancer_classifier/sample_output/cropped_images'
#BSSA_im_fp = os.path.join(DATADIR, 'test_ims_master/test_ims_mini/size_matched/cropped_images_sf_matched'

#ix = int(len(NYU_ims)/4)
#for _ in range(ix):
#    for view in VIEWS.LIST:
#        BSSA_im_fp = random.choice([im for im in BSSA_ims if view in im])
#        bim = loading.load_image(
#                    image_path=BSSA_im_fp,
#                    view=view,
#                    horizontal_flip='NO'
#                    )
#        
#        NYU_im_fp = random.choice([im for im in NYU_ims if view.replace('-','_') in im])
#        nim = loading.load_image(
#                image_path=NYU_im_fp,
#                view=view,
#                horizontal_flip='NO'
#                )
#            
#        show_2(bim, nim, title='Different image sizes (to scale)', stitle1='BSSA - {} pixels'.format(bim.shape), stitle2='NYU - {} pixels'.format(str(nim.shape)))
#%%
for nyu, bssa in zip(NYU_exams, BSSA_exams):
#    print(nyu, bssa)
    for view in VIEWS.LIST:
        nim = loading.load_image(
            image_path=os.path.join(NYU_im_fp, nyu[view][0] + '.png'),
            view=view,
            horizontal_flip=nyu['horizontal_flip']
            )
        bim = loading.load_image(
            image_path=os.path.join(BSSA_im_fp, bssa[view][0] + '.hdf5'),
            view=view,
            horizontal_flip=nyu['horizontal_flip']
            )
        
        print(NYU_im_fp, ':')
#        show(nim, figsize=(10,10))
        print(BSSA_im_fp, ':')
#        show(bim, figsize=(10,10))
        title = 'Cropped breast, image sizes NYU vs BSSA'
        if nim.shape[0] > bim.shape[0]:
            show_2(bim, nim, title=title, stitle1='SA. size: '+str(bim.shape), stitle2='NYU. size: '+str(nim.shape), figsize=(10,10))
        else:
            show_2(nim, bim, title=title, stitle1='NYU. size: '+str(nim.shape), stitle2='SA. size: '+str(bim.shape), figsize=(10,10))
        input()
#import IPython; IPython.embed()   
#    if i == 2:
#        break
    


