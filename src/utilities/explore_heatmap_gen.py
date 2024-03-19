# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import matplotlib.pyplot as plt
from src.utilities.pickling import unpickle_from_file
import src.heatmaps.models as models

patch_model_path = '/media/2TB_HDD/mydev/breast_cancer_classifier/models/sample_patch_model.p'
#minibatch_x = unpickle_from_file('/home/mlim-user/Documents/james/my_dev/breast_cancer_classifier/sample_output/0_L_CC_minibatch.pkl')

#patches = unpickle_from_file('/home/mlim-user/Documents/james/my_dev/breast_cancer_classifier/sample_output/1_L_MLO_patches.pkl')
patches = unpickle_from_file('/media/2TB_HDD/mydev/breast_cancer_classifier/sample_output/1_L_MLO_patches.pkl')


# for i, batch in enumerate(patches):
#     plt.imshow(batch[:,:,0], cmap='gray')
#     plt.show()
#     input()

model = models.ModifiedDenseNet121(num_classes=4)

model.load_from_path(patch_model_path)
