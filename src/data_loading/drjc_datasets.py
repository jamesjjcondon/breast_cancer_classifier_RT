#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:17:03 2020

@author: James Condon
"""
# Modified from and based on:

# Copyright (C) 2019 Nan Wu, Jason Phang, Jungkyu Park, Yiqiu Shen, Zhe Huang, Masha Zorin, 
#   Stanisław Jastrzębski, Thibault Févry, Joe Katsnelson, Eric Kim, Stacey Wolfson, Ujas Parikh, 
#   Sushma Gaddam, Leng Leng Young Lin, Kara Ho, Joshua D. Weinstein, Beatriu Reig, Yiming Gao, 
#   Hildegard Toth, Kristine Pysarenko, Alana Lewin, Jiyon Lee, Krystal Airola, Eralda Mema, 
#   Stephanie Chung, Esther Hwang, Naziya Samreen, S. Gene Kim, Laura Heacock, Linda Moy, 
#   Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of a modified version of breast_cancer_classifier:
# https://github.com/nyukat/breast_cancer_classifier
# Wu N, Phang J, Park J et al. Deep neural networks improve radiologists’ performance in breast cancer screening.
# PubMed - NCBI [Internet]. [cited 2020 Mar 6]. Available from: https://www.ncbi.nlm.nih.gov/pubmed/31603772
#
# breast_cancer_classifier is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# breast_cancer_classifier is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with breast_cancer_classifier.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from src.utilities import pickling
from src.data_loading import loading
from src.constants import VIEWS


class BSSA_exams(Dataset):
    """ Pytorch dataset for 4-view exams from nyu-style exam_list 
    NB:
        - each filename looks like
            I11958-E16-A177135-S-i1-R-MLO_p0e0i0.hdf5
            where p0 == patient never had cancer
            e0 == screening episode is cancer free
            i0 == image is cancer free 
    """
    def __init__(self, exam_list_in, transform, parameters, train_val_or_test='train'):
        super(BSSA_exams, self).__init__()
        if type(exam_list_in) == list:
            exam_list = exam_list_in
        elif type(exam_list_in) == str:
            exam_list = pickling.unpickle_from_file(exam_list_in)
        else:
            raise ValueError("exam_list_in must be either a str file_path or loaded list")
        
        self.exam_list = np.array(exam_list)
        #print('\nlen exam_list:', len(self.exam_list))
        self.transform = transform
        self.random_number_generator = np.random.RandomState(parameters.seed)
        self.image_extension = ".hdf5" if parameters.use_hdf5 else ".png"
        if train_val_or_test == 'train':
            self.im_path = parameters.train_image_path
            self.hms_path = parameters.train_heatmaps_path
        elif train_val_or_test == 'val':
            self.im_path = parameters.val_image_path
            self.hms_path = parameters.val_heatmaps_path
        self.using_heatmaps = parameters.use_heatmaps
        self.augmenting = parameters.augmentation
        self.mcn = (parameters.max_crop_noise, parameters.max_crop_noise)
        self.mcsn = parameters.max_crop_size_noise
        # create episode-wise target list based on filename (episode code same for all images in an episode):
        self.episode_wise_targets = torch.tensor([0 if 'e0' in x['L-CC'][0] else 1 for x in self.exam_list])        
        # print('len targets:', len(self.episode_wise_targets))

        self.ca_eps = len([x for x in self.exam_list if 'e1' in x['L-CC'][0]])
        self.ca_free_eps = len([x for x in self.exam_list if 'e0' in x['L-CC'][0]])
        assert self.ca_eps + self.ca_free_eps == len(self.exam_list)
        # print('\n\tTotal {} n_counts:'.format(train_val_or_test))
        # print('\tCancer_episodes:', self.ca_eps)
        # print('\tCancer_free_episodes:', self.ca_free_eps)
        
    def __getitem__(self, index):
        """
        Return tensor dict of all 4 views from one exam in nyu-style exam_list
        """
        datum = self.exam_list[index]
        #BSSA_ID_pref = datum['L-CC'][0].rsplit('-', 3)[0] # take start of one filename (eg 'I777888-E2-A102778-S-i6') for logging 
        batch_dict = {view: [] for view in VIEWS.LIST} # {'L-CC': [], 'R-CC': [], 'L-MLO': [], 'R-MLO': []}
        self.loaded_image_dict = {view: [] for view in VIEWS.LIST}
        loaded_heatmaps_dict = {view: [] for view in VIEWS.LIST}
        targets_dict = {view: {'ca':[], 'benign':[]} for view in VIEWS.LIST}
        for view in VIEWS.LIST: # [L_CC, R_CC, L_MLO, R_MLO]
            for short_file_path in datum[view]:
                loaded_image, attrs = loading.load_h5_image_and_attrs(
                    image_path=os.path.join(self.im_path, short_file_path + self.image_extension),
                    view=view,
                    horizontal_flip=datum["horizontal_flip"])
                if self.using_heatmaps:
                    loaded_heatmaps = loading.load_heatmaps(
                        benign_heatmap_path=os.path.join(self.hms_path, "heatmap_benign",
                                                         short_file_path + ".hdf5"),
                        malignant_heatmap_path=os.path.join(self.hms_path, "heatmap_malignant",
                                                            short_file_path + ".hdf5"),
                        view=view,
                        horizontal_flip=datum["horizontal_flip"])
                else:
                    loaded_heatmaps = None

                self.loaded_image_dict[view].append(loaded_image)
                loaded_heatmaps_dict[view].append(loaded_heatmaps)     
        for view in VIEWS.LIST:
            image_index = 0
            if self.augmenting:
                image_index = self.random_number_generator.randint(low=0, high=len(datum[view]))
            
            cropped_image, cropped_heatmaps = loading.augment_and_normalize_image(
                image=self.loaded_image_dict[view][image_index],
                auxiliary_image=loaded_heatmaps_dict[view][image_index],
                view=view,
                best_center=datum["best_center"][view][image_index],
                random_number_generator=self.random_number_generator,
                augmentation=self.augmenting,
                max_crop_noise=self.mcn,
                max_crop_size_noise=self.mcsn
                )
            if loaded_heatmaps_dict[view][image_index] is None:
                batch_dict[view].append(cropped_image[:, :, np.newaxis])
            else:
                batch_dict[view].append(np.concatenate([
                    cropped_image[:, :, np.newaxis],
                    cropped_heatmaps], axis=2))
            if VIEWS.is_left(view):
                targets_dict[view]['ca'] = attrs['Left_views_ca']
                targets_dict[view]['benign'] = attrs['Left_views_benign']
            elif VIEWS.is_right(view):
                targets_dict[view]['ca'] = attrs['Right_views_ca']
                targets_dict[view]['benign'] = attrs['Right_views_benign']               
        if self.transform is not None: # only one cropped im/heatmap gets added to batch_dict per view (regardless of extra ims / view)
            batch_dict = {view: self.transform(batch_dict[view][0]) for view in VIEWS.LIST} # so index is always 0
        return batch_dict, targets_dict

    def __len__(self):
        return len(self.exam_list)
