#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 10:00:38 2020

Takes original NYU-style data.pkl dictionary and halves ideal centre x and y
to match NYU augmentations. 
Visualises results. 

@author: james condon
"""
import os
import numpy as np
from argparse import ArgumentParser
from src.utilities.vis_heatmaps import vis_heatmaps
from src.utilities.pickling import unpickle_from_file, pickle_to_file
from src.constants import DATADIR, VIEWS

def main(params):
    
    data_sf0 = unpickle_from_file(
            os.path.join(DATADIR, 'test_ims_master/data_sf0.pkl'))
    
    # load 2 copies
    data_sf2_NYU_ideal_centre, data_sf2 = unpickle_from_file(
            os.path.join(DATADIR, 'test_ims_master/data_sf2.pkl')), unpickle_from_file(
            os.path.join(DATADIR, 'test_ims_master/data_sf2.pkl'))
        
    # Make sure filenames are all in the same order:
    for d1, d2 in zip(data_sf0, data_sf2):
        for view in VIEWS.LIST:
            print(d1[view])
            assert d1[view] == d2[view]
    
    # change ideal centres in data_sf2_NYU_ideal_centre to half of values from full-scale images to match NYU pipe: 
    for d1, d2 in zip(data_sf0, data_sf2_NYU_ideal_centre):
        for view in VIEWS.LIST:
            for ix in range(len(d1['best_center'][view])):
                print(ix, '\nview:', view)
                print('\toriginal ideal_centre:', d1['best_center'][view][ix])   
                sf2_NYU_ideal_centre = tuple([int(np.round(x/2)) for x in d1['best_center'][view][ix]])
                print('\t sf2 ideal_centre:', sf2_NYU_ideal_centre)
                d2['best_center'][view][ix] = tuple(sf2_NYU_ideal_centre)
    
    
    # compare visually        
    params.data_in = data_sf2
    # loads sf2 during __init__: self.ds
    vis = vis_heatmaps(params)
    vis.vis_centers(AN=2152823) #n=40) #AN=114173) 2152823
    
    params.data_in = data_sf2_NYU_ideal_centre
    vis = vis_heatmaps(params)
    vis.vis_centers(AN=2152823) #AN=114173)

    # save:
#    pickle_to_file(
#            os.path.join(XHD, 'test_ims_master/data_sf2_NYU_IC.pkl'),
#            data_sf2_NYU_ideal_centre)
    
if __name__== "__main__":
    
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--augmentation', type=str, default=True)
#    parser.add_argument('--data-fp', type=str, 
#                        default=os.path.join(XHD, 'test_ims_master/data_sf2.pkl'))
    parser.add_argument('--infer_fp', default=False)
    parser.add_argument('--max_crop_noise', type=int, default=100)
    parser.add_argument('--max_crop_size_noise', type=int, default=100)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--train_heatmaps_path', type=str, 
                        default=os.path.join(XHD, 'test_ims_master/heatmaps_sf2'))
    parser.add_argument('--train_image_path', type=str, 
                        default=os.path.join(XHD, 'test_ims_master/cropped_images_sf2'))
    parser.add_argument('--use_heatmaps', action='store_true', default=True)
    parser.add_argument('--use_hdf5', action='store_true', default=True)
    parser.add_argument('--use_n_exams', default=False)

    params = parser.parse_args()
    
    main(params)