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
import math
from src.utilities import pickling, reading_images
from src.utilities.all_utils import show_2 #, show
from src.data_loading import loading
from src.data_loading.augmentations import simple_resize
from src.constants import DATADIR, NVMEDIR, VIEWS, REPODIR

#DATADIR = '/nvme/james'

NYU_im_fp = os.path.join(REPODIR, 'sample_data/images')
NYU_ims = sorted(os.listdir(NYU_im_fp))
NYU_exams = pickling.unpickle_from_file(os.path.join(REPODIR, 'sample_data/exam_list_after_centres.pkl'))

#BSSA_im_fp = os.path.join(DATADIR, 'test_ims_master/cropped_images_sf2') #DATADIR, 'test_ims_master/renamed_dicoms')
BSSA_im_fp = '/data/james/NYU_Retrain/test_ims_master/test_ims_mini/renamed_dicoms_mini'
#BSSA_im_fp = '/data/james/NYU_Retrain/test_ims_master/test_ims_mini/renamed_dicoms_mini'

BSSA_ims = [x for x in sorted(os.listdir(BSSA_im_fp)) if x.endswith('.dcm')]

#%%
all_BSSA_exams = pickling.unpickle_from_file(
    os.path.join(DATADIR, 'test_ims_master/test_ims_mini/cropped_exam_list_sf_matched.pkl'))
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

#%%

NYU_ims = get_files_by_file_size(NYU_im_fp, reverse=True)
BSSA_ims = get_files_by_file_size(BSSA_im_fp, reverse=True)#[:len(NYU_ims)]

#NYU_im_fp = '/home/mlim-user/Documents/james/my_dev/nyukat2.0/sample_data/images/3_R_MLO.png'
#BSSA_im_fp = '/data/james/NYU_retrain/test_ims_master/renamed_dicoms/I230097-E11-A459999-S-i3-R-MLO_p1e1i1.dcm'

ix = int(len(NYU_ims)/4)
for i in range(ix):
    for view in VIEWS.LIST:
        BSSA_im_fp = random.choice([im for im in BSSA_ims if view in im])
        bim = loading.load_image(
                    image_path=BSSA_im_fp,
                    view=view,
                    horizontal_flip='NO'
                    )
        
        NYU_im_fp = random.choice([im for im in NYU_ims if view.replace('-','_') in im])
        nim = loading.load_image(
                image_path=NYU_im_fp,
                view=view,
                horizontal_flip='NO'
                )
        
        h, w = nim.shape
        hwr = h / w
        #bim = simple_resize(bim, size=(h,math.ceil(h/hwr)))
        fname1 = os.path.join(REPODIR, 'figures_and_tables/im_size_dif/' + str(i) + '_' + view + '_NYU')
        fname2 = os.path.join(REPODIR, 'figures_and_tables/im_size_dif/' + str(i) + '_' + view + '_BSSA')
        
        if i == 0:
            if view == 'MLO':
            print(nim.shape)
            print(bim.shape)
        show_2(nim, bim, title=None, save=False, fp1=fname1, fp2=fname2) #, stitle1='BSSA', stitle2='NYU') #, labels=False, ticks=False)

        # plt.figure()
        # # plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
        # # plt.xticks([])
        # # plt.yticks([])
        # plt.imshow(bim, cmap='gray')
        # plt.savefig(fname=fname1, dpi=600)
        # plt.show()
        # plt.clf()
        
        # plt.figure()
        # # plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
        # # plt.xticks([])
        # # plt.yticks([])
        # plt.imshow(nim, cmap='gray')
        # plt.savefig(fname=fname2 dpi=600)
        # plt.show()
        #print('continue?')
        #input()