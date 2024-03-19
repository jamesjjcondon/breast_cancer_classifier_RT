#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import math
import pandas as pd
import random
from multiprocessing import Pool
import argparse
from functools import partial
import scipy.ndimage
import numpy as np
#from IPython import embed
from src.data_loading.augmentations import simple_resize
import src.utilities.pickling as pickling
import src.utilities.reading_images as reading_images
import src.utilities.saving_images as saving_images
import src.utilities.data_handling as data_handling
from src.constants import BASECOLS2, DATADIR
#%%
def get_masks_and_sizes_of_connected_components(img_mask):
    """
    Finds the connected components from the mask of the image
    """
    mask, num_labels = scipy.ndimage.label(img_mask)

    mask_pixels_dict = {}
    for i in range(num_labels+1):
        this_mask = (mask == i)
        if img_mask[this_mask][0] != 0:
            # Exclude the 0-valued mask
            mask_pixels_dict[i] = np.sum(this_mask)
        
    return mask, mask_pixels_dict


def get_mask_of_largest_connected_component(img_mask):
    """
    Finds the largest connected component from the mask of the image
    """
    mask, mask_pixels_dict = get_masks_and_sizes_of_connected_components(img_mask)
    largest_mask_index = pd.Series(mask_pixels_dict).idxmax()
    largest_mask = mask == largest_mask_index
    return largest_mask


def get_edge_values(img, largest_mask, axis):
    """
    Finds the bounding box for the largest connected component
    """
    assert axis in ["x", "y"]
    has_value = np.any(largest_mask, axis=int(axis == "y"))
    edge_start = np.arange(img.shape[int(axis == "x")])[has_value][0]
    edge_end = np.arange(img.shape[int(axis == "x")])[has_value][-1] + 1
    return edge_start, edge_end


def get_bottommost_pixels(img, largest_mask, y_edge_bottom):
    """
    Gets the bottommost nonzero pixels of dilated mask before cropping. 
    """
    bottommost_nonzero_y = y_edge_bottom - 1
    bottommost_nonzero_x = np.arange(img.shape[1])[largest_mask[bottommost_nonzero_y, :] > 0]
    return bottommost_nonzero_y, bottommost_nonzero_x


def get_distance_from_starting_side(img, mode, x_edge_left, x_edge_right):
    """
    If we fail to recover the original shape as a result of erosion-dilation 
    on the side where the breast starts to appear in the image, 
    we record this information.
    """
    if mode == "left":
        return img.shape[1] - x_edge_right
    else:
        return x_edge_left


def include_buffer_y_axis(img, y_edge_top, y_edge_bottom, buffer_size):
    """
    Includes buffer in all sides of the image in y-direction
    """
    if y_edge_top > 0:
        y_edge_top -= min(y_edge_top, buffer_size)
    if y_edge_bottom < img.shape[0]:
        y_edge_bottom += min(img.shape[0] - y_edge_bottom, buffer_size)
    return y_edge_top, y_edge_bottom     


def include_buffer_x_axis(img, mode, x_edge_left, x_edge_right, buffer_size):
    """
    Includes buffer in only one side of the image in x-direction
    """
    if mode == "left":
        if x_edge_left > 0:
            x_edge_left -= min(x_edge_left, buffer_size)
    else:
        if x_edge_right < img.shape[1]:
            x_edge_right += min(img.shape[1] - x_edge_right, buffer_size)
    return x_edge_left, x_edge_right


def convert_bottommost_pixels_wrt_cropped_image(mode, bottommost_nonzero_y, bottommost_nonzero_x,
                                                y_edge_top, x_edge_right, x_edge_left):
    """
    Once the image is cropped, adjusts the bottommost pixel values which was originally w.r.t. the original image
    """
    bottommost_nonzero_y -= y_edge_top
    if mode == "left":
        bottommost_nonzero_x = x_edge_right - bottommost_nonzero_x  # in this case, not in sorted order anymore.
        bottommost_nonzero_x = np.flip(bottommost_nonzero_x, 0)
    else:
        bottommost_nonzero_x -= x_edge_left
    return bottommost_nonzero_y, bottommost_nonzero_x


def get_rightmost_pixels_wrt_cropped_image(mode, largest_mask_cropped, find_rightmost_from_ratio):
    """
    Ignores top find_rightmost_from_ratio of the image and searches the rightmost nonzero pixels
    of the dilated mask from the bottom portion of the image.
    """
    ignore_height = int(largest_mask_cropped.shape[0] * find_rightmost_from_ratio)
    rightmost_pixel_search_area = largest_mask_cropped[ignore_height:, :]
    rightmost_pixel_search_area_has_value = np.any(rightmost_pixel_search_area, axis=0)
    rightmost_nonzero_x = np.arange(rightmost_pixel_search_area.shape[1])[
        rightmost_pixel_search_area_has_value][-1 if mode == 'right' else 0]
    rightmost_nonzero_y = np.arange(rightmost_pixel_search_area.shape[0])[
        rightmost_pixel_search_area[:, rightmost_nonzero_x] > 0] + ignore_height

    # rightmost pixels are already found w.r.t. newly cropped image, except that we still need to
    #   reflect horizontal_flip
    if mode == "left":
        rightmost_nonzero_x = largest_mask_cropped.shape[1] - rightmost_nonzero_x
        
    return rightmost_nonzero_y, rightmost_nonzero_x


def crop_img_from_largest_connected(img, mode, erode_dialate=True, iterations=100, 
                                    buffer_size=50, find_rightmost_from_ratio=1/3):
    """
    Performs erosion on the mask of the image, selects largest connected component,
    dialates the largest connected component, and draws a bounding box for the result
    with buffers

    input:
        - img:   2D numpy array
        - mode:  breast pointing left or right

    output: a tuple of (window_location, rightmost_points, 
                        bottommost_points, distance_from_starting_side)
        - window_location: location of cropping window w.r.t. original dicom image so that segmentation
           map can be cropped in the same way for training.
        - rightmost_points: rightmost nonzero pixels after correctly being flipped in the format of 
                            ((y_start, y_end), x)
        - bottommost_points: bottommost nonzero pixels after correctly being flipped in the format of
                             (y, (x_start, x_end))
        - distance_from_starting_side: number of zero columns between the start of the image and start of
           the largest connected component w.r.t. original dicom image.
    """
    assert mode in ("left", "right")

    img_mask = img > 0

    # Erosion in order to remove thin lines in the background
    if erode_dialate:
        img_mask = scipy.ndimage.morphology.binary_erosion(img_mask, iterations=iterations)

    # Select mask for largest connected component
    largest_mask = get_mask_of_largest_connected_component(img_mask)

    # Dilation to recover the original mask, excluding the thin lines
    if erode_dialate:
        largest_mask = scipy.ndimage.morphology.binary_dilation(largest_mask, iterations=iterations)
    
    # figure out where to crop
    y_edge_top, y_edge_bottom = get_edge_values(img, largest_mask, "y")
    x_edge_left, x_edge_right = get_edge_values(img, largest_mask, "x")

    # extract bottommost pixel info
    bottommost_nonzero_y, bottommost_nonzero_x = get_bottommost_pixels(img, largest_mask, y_edge_bottom)

    # include maximum 'buffer_size' more pixels on both sides just to make sure we don't miss anything
    y_edge_top, y_edge_bottom = include_buffer_y_axis(img, y_edge_top, y_edge_bottom, buffer_size)
    
    # If cropped image not starting from corresponding edge, they are wrong. Record the distance, will reject if not 0.
    distance_from_starting_side = get_distance_from_starting_side(img, mode, x_edge_left, x_edge_right)

    # include more pixels on either side just to make sure we don't miss anything, if the next column
    #   contains non-zero value and isn't noise
    x_edge_left, x_edge_right = include_buffer_x_axis(img, mode, x_edge_left, x_edge_right, buffer_size)

    # convert bottommost pixel locations w.r.t. newly cropped image. Flip if necessary.
    bottommost_nonzero_y, bottommost_nonzero_x = convert_bottommost_pixels_wrt_cropped_image(
        mode,
        bottommost_nonzero_y,
        bottommost_nonzero_x,
        y_edge_top,
        x_edge_right,
        x_edge_left
    )

    # calculate rightmost point from bottom portion of the image w.r.t. cropped image. Flip if necessary.
    rightmost_nonzero_y, rightmost_nonzero_x = get_rightmost_pixels_wrt_cropped_image(
        mode,
        largest_mask[y_edge_top: y_edge_bottom, x_edge_left: x_edge_right],
        find_rightmost_from_ratio
    )

    # save window location in medical mode, but everything else in training mode
    return (y_edge_top, y_edge_bottom, x_edge_left, x_edge_right), \
        ((rightmost_nonzero_y[0], rightmost_nonzero_y[-1]), rightmost_nonzero_x), \
        (bottommost_nonzero_y, (bottommost_nonzero_x[0], bottommost_nonzero_x[-1])), \
        distance_from_starting_side


def image_orientation(horizontal_flip, side):
    """
    Returns the direction where the breast should be facing in the original image
    This information is used in cropping.crop_img_horizontally_from_largest_connected
    """
    assert horizontal_flip in ['YES', 'NO'], "Wrong horizontal flip"
    assert side in ['L', 'R'], "Wrong side"
    if horizontal_flip == 'YES':
        if side == 'R':
            return 'right'
        else:
            return 'left'
    else:
        if side == 'R':
            return 'left'
        else:
            return 'right'


def crop_mammogram(input_data_folder, exam_list_path, cropped_exam_list_path, output_data_folder,
                   num_processes, num_iterations, buffer_size, shrink_factor, contrast_window, re_training, df_fp=None):
    """
    In parallel, crops mammograms in DICOM format found in input_data_folder and save as png format in
    output_data_folder and saves new image list in cropped_image_list_path
    """
    exam_list = pickling.unpickle_from_file(exam_list_path)
    
    image_list = data_handling.unpack_exam_into_images(exam_list)
    
    #embed()
    if re_training: # load ground-truth labels metadata to match each image (left and right benign and cancer binaries):
        df = pd.read_csv(df_fp)[BASECOLS2]
    else:
        df = None
    if os.path.exists(output_data_folder):
        # Prevent overwriting to an existing directory
        print("Error: the directory to save cropped images \n {} \nalready exists. Are you sure you want to continue? (enter)".format(output_data_folder))
        input()
    else:
        os.makedirs(output_data_folder)

    crop_mammogram_one_image_func = partial(
        crop_mammogram_one_image_short_path,
        input_data_folder=input_data_folder, 
        output_data_folder=output_data_folder,
        num_iterations=num_iterations,
        buffer_size=buffer_size,
        shrink_factor=shrink_factor,
        contrast_window=contrast_window,
        df=df,
        re_training=re_training
    )
    with Pool(num_processes) as pool:
        cropped_image_info = pool.map(crop_mammogram_one_image_func, image_list)
    
    window_location_dict = dict([x[0] for x in cropped_image_info])
    rightmost_points_dict = dict([x[1] for x in cropped_image_info])
    bottommost_points_dict = dict([x[2] for x in cropped_image_info])
    distance_from_starting_side_dict = dict([x[3] for x in cropped_image_info])

    data_handling.add_metadata(exam_list, "window_location", window_location_dict)
    data_handling.add_metadata(exam_list, "rightmost_points", rightmost_points_dict)
    data_handling.add_metadata(exam_list, "bottommost_points", bottommost_points_dict)
    data_handling.add_metadata(exam_list, "distance_from_starting_side", distance_from_starting_side_dict)
    
    pickling.pickle_to_file(cropped_exam_list_path, exam_list)
    

def crop_mammogram_one_image(scan, input_file_path, output_file_path, num_iterations, buffer_size, re_training, contrast_window, shrink_factor=None, gt_data=None):
    """
    Crops a mammogram, saves as png file, includes the following additional information:
        - window_location: location of cropping window w.r.t. original dicom image so that segmentation
           map can be cropped in the same way for training.
        - rightmost_points: rightmost nonzero pixels after correctly being flipped
        - bottommost_points: bottommost nonzero pixels after correctly being flipped
        - distance_from_starting_side: number of zero columns between the start of the image and start of
           the largest connected component w.r.t. original dicom image.
    
    For domain transfer to BSSA images, and fidelity of real-world, biological malignancy features, we preserve aspect-ratio of images
    by dividing height and weight equally. 
    """
    if shrink_factor is not None and not False and not 0 and not 1:
        assert shrink_factor < 5
    #embed()
    NYU_sizes = [
            (4084, 3328), # sizes per github.com/nyukat/breast_cancer_classifier
           (3328, 2560),
           (3328, 2560),
           (4096, 3328)
            ]
    
    image = reading_images.read_image_dcm(
            input_file_path, 
            window=contrast_window)
    
    if shrink_factor not in [None, False, 0, 1, 'rel']:
#        print('\tshrinking image...')
        
        # downscale whole original image height to model window 
        if shrink_factor == 'window': 
            print('\n\t downscaling whole original image height to model window...')
            NYU_MLO = (2974, 1748)
            NYU_CC = (2677, 1943)
            if 'MLO' in scan['view']:
                shrink_factor = image.shape[0] / NYU_MLO[0]
            elif 'CC' in scan['view']:
                shrink_factor = image.shape[0] / NYU_CC[0]
                       
        elif shrink_factor == 'NYU_large':
            shrink_factor = image.shape[0] / 4096 # largest height
        elif shrink_factor == 'NYU_med':
            shrink_factor = image.shape[0] / 4084 # largest height
        elif shrink_factor == 'NYU_small':
            shrink_factor = image.shape[0] / 3328 # largest height

#        print('shrink_factor:', shrink_factor)

        image = simple_resize(image, tuple(math.ceil(x/shrink_factor) for x in image.shape))              
        
#        print('\noutput image size:', image.shape, '\n')
        
        assert image.min() == 0 # standardise to keep min-zero and fit cropping pipe

    try:
        # error detection using erosion. Also get cropping information for this image.
        cropping_info = crop_img_from_largest_connected(
            image, 
            image_orientation(scan['horizontal_flip'], scan['side']), 
            True, 
            num_iterations, 
            buffer_size, 
            1/3
        )
        # could add r and lf ca and benign info here
    except Exception as error:
        print(input_file_path, "\n\tFailed to crop image because image is invalid.", str(error))
    else:
        
        top, bottom, left, right = cropping_info[0]

        target_parent_dir = os.path.split(output_file_path)[0]
        if not os.path.exists(target_parent_dir):
            os.makedirs(target_parent_dir)
        
        if re_training:
            assert gt_data is not None
            if gt_data.shape[0] == 2:
                gt_data.drop_duplicates(inplace=True) # remove duplicated episode info
            try:
#                print('\n\tsaving image and attributes')
                saving_images.save_image_and_attrs_hdf5(
                        image[top:bottom, left:right], output_file_path,
                        data=gt_data
                        )
            except Exception as error:
                print(input_file_path, "\n\tError while saving image.", str(error))
                print(gt_data, gt_data.shape, type(gt_data))
                print(gt_data['new_names'])
                print(scan['short_file_path'])
                print(scan)
                raise ValueError
        else:
            try:
#                print('\n\tsaving only image')
                saving_images.save_image_as_hdf5(image[top:bottom, left:right], output_file_path)
            except Exception as error:
                print(input_file_path, "\n\tError while saving image.", str(error))

    return cropping_info


def crop_mammogram_one_image_short_path(scan, input_data_folder, output_data_folder,
                                        num_iterations, buffer_size, shrink_factor, contrast_window, df, re_training):
    """
    Crops a mammogram from a short_file_path

    See: crop_mammogram_one_image
    """
    full_input_file_path = os.path.join(input_data_folder, scan['short_file_path']+'.dcm')
    full_output_file_path = os.path.join(output_data_folder, scan['short_file_path'] + '.hdf5')
    
    if re_training:
        gt_data = df[df['new_names'] == scan['short_file_path']+'.dcm']
        assert gt_data.shape[0] > 0 
        assert gt_data.shape[1] > 0
    else:
        gt_data = None
#    if os.path.exists(full_output_file_path):
#        print(scan['short_file_path'], 'already exists. Skipping')
#        return
    cropping_info = crop_mammogram_one_image(
        scan=scan,
        input_file_path=full_input_file_path,
        output_file_path=full_output_file_path,
        num_iterations=num_iterations,
        buffer_size=buffer_size,
        shrink_factor=shrink_factor,
        contrast_window=contrast_window,
        gt_data=gt_data,
        re_training=re_training
    )
    return list(zip([scan['short_file_path']] * 4, cropping_info))


#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove background of image and save cropped files')
#    parser.add_argument('--input-data-folder', required=True)
#    parser.add_argument('--output-data-folder', required=True)
#    parser.add_argument('--exam-list-path', required=True)
#    parser.add_argument('--cropped-exam-list-path', required=True)
    parser.add_argument('--input-data-folder', required=False)
    parser.add_argument('--output-data-folder', required=False)
    parser.add_argument('--exam-list-path', required=False)
    parser.add_argument('--cropped-exam-list-path', required=False)
    
    parser.add_argument('--num-processes', default=1, type=int)
    parser.add_argument('--num-iterations', default=100, type=int)
    parser.add_argument('--buffer-size', default=50, type=int)
    parser.add_argument('--shrink-factor', default=2)
    parser.add_argument('--contrast_window', default='Normal')
    parser.add_argument('--re-training', action='store_true')
    parser.add_argument('--df-fp', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.re_training:
        assert args.df_fp is not None, "If re-training, enter full file path of csv with image ground truth (Left and Right cancer and benign binaries)"
        assert args.df_fp.endswith('.csv'), "only .csv currently supported"
    print('\nargs.exam_list_path:', args.exam_list_path)
    print('\nargs.cropped_exam_list_path:', args.cropped_exam_list_path)
    
    print('\nImages will be shrunk by a factor of {} before cropping'.format(
            args.shrink_factor))
    
    print('\n images to be saved with attributes?/re-training?.')
    print(args.re_training)
    
    crop_mammogram(
        input_data_folder=args.input_data_folder, 
        exam_list_path=args.exam_list_path, 
        cropped_exam_list_path=args.cropped_exam_list_path, 
        output_data_folder=args.output_data_folder, 
        num_processes=args.num_processes,
        num_iterations=args.num_iterations,
        buffer_size=args.buffer_size,
        shrink_factor=args.shrink_factor,
        contrast_window=args.contrast_window,
        re_training=args.re_training,
        df_fp=args.df_fp
    )
    
#    crop_mammogram(
#        input_data_folder=os.path.join(DATADIR, 'test_ims_master/renamed_dicoms'), 
#        exam_list_path=os.path.join(DATADIR, 'test_ims_master/pre_crop_exam_list.pkl'), 
#        cropped_exam_list_path=os.path.join(DATADIR, 'test_ims_master/large_matched/cropped_exam_list_sf_matched_large.pkl'), 
#        output_data_folder=os.path.join(DATADIR, 'test_ims_master/large_matched/cropped_images_sf_matched_large'), 
#        num_processes=12,
#        num_iterations=args.num_iterations,
#        buffer_size=args.buffer_size,
#        shrink_factor='match', #args.shrink_factor,z
#        contrast_window='Normal',
#        re_training=args.re_training,
#        df_fp=args.df_fp
#    )