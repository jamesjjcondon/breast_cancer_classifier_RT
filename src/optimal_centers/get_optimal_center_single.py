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
"""
Runs search_windows_and_centers.py and extract_centers.py in the same directory
"""
import argparse
import os
import numpy as np
from src.constants import half as INPUT_SIZE_DICT
import matplotlib.pyplot as plt
import src.utilities.pickling as pickling
import src.utilities.reading_images as reading_images
import src.utilities.data_handling as data_handling
from src.data_loading import loading
from src.optimal_centers import calc_optimal_centers
import src.optimal_centers.get_optimal_centers as get_optimal_centers

#%%

def get_optimal_center_single(cropped_mammogram_path, metadata): #metadata_path):
    """
    Get optimal center for single example
    """
#    metadata = pickling.unpickle_from_file(metadata_path)
    image = reading_images.read_image_mat(cropped_mammogram_path)
    optimal_center = get_optimal_centers.extract_center(metadata, image)
    metadata["best_center"] = optimal_center
#    pickling.pickle_to_file(metadata_path, metadata)
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.scatter(optimal_center[0], optimal_center[1], s=30, c='r')
    plt.show()
    return optimal_center

def main():
    cropped_exam_list_path='/data/james/NYU_exval/images_5050/cropped_exam_list.pkl'
    data_prefix = '/data/james/NYU_exval/images_5050/cropped_images'
    exam_list = pickling.unpickle_from_file(cropped_exam_list_path)
    data_list = data_handling.unpack_exam_into_images(exam_list, cropped=True)
    image_extension = '.hdf5'
    n = 10
    for data in data_list[:n]:
        fp = os.path.join(data_prefix, data['short_file_path'] + image_extension)
        image = reading_images.read_image_mat(fp)
        
        # get_optimal_centers.extract_center:
        datum = data
        image = loading.flip_image(image, datum["full_view"], datum['horizontal_flip'])
        plt.figure()
        plt.imshow(image)
        x = datum["rightmost_points"][1], datum["rightmost_points"][1]
        y = datum['rightmost_points'][0]
        plt.scatter(x, y, s=30, c='r')
        plt.show()
        if datum["view"] == "MLO":
            print('using MLO path')
            tl_br_constraint = calc_optimal_centers.get_bottomrightmost_pixel_constraint(
                rightmost_x=datum["rightmost_points"][1],
                bottommost_y=datum["bottommost_points"][0],
            )
        elif datum["view"] == "CC":
            print('using CC path')
            tl_br_constraint = calc_optimal_centers.get_rightmost_pixel_constraint(
                rightmost_x=datum["rightmost_points"][1]
            )
        else:
            raise RuntimeError(datum["view"])        
        
        optimal_center = calc_optimal_centers.get_image_optimal_window_info(
                image,
                com=np.array(image.shape) // 2,
                window_dim=np.array(INPUT_SIZE_DICT[datum["full_view"]]),
                tl_br_constraint=tl_br_constraint
                )
        plt.figure()
        plt.imshow(image)
        plt.scatter(optimal_center['best_center_x'], optimal_center['best_center_y'], s=30, c='r')
        plt.show()
        
        com = np.array(image.shape) // 2
        # window_dim is INPUT_SIZE_DICT
        window_dim = np.array(INPUT_SIZE_DICT[datum["full_view"]])

        # calc_optimal_centers.get_image_optimal_window_info()
        image_dim = image.shape
        if cumsum is None:
            cumsum = calc_optimal_centers.get_image_cumsum(image)
            
        window_area = np.prod(window_dim)
        
        tl, br = calc_optimal_centers.get_candidate_center_topleft_bottomright(
            com=com, image_dim=image_dim, window_dim=window_dim, step=1)
        
        if tl_br_constraint:
            tl, br = tl_br_constraint(tl=tl, br=br, image=image, window_dim=window_dim)
        
        y_grid_axis = np.arange(tl[0], br[0], 1)
        x_grid_axis = np.arange(tl[1], br[1], 1)
        window_center_ls = calc_optimal_centers.get_joint_axes(y_grid_axis, x_grid_axis)
    
        tl_ls, br_ls = calc_optimal_centers.get_candidate_topleft_bottomright(
            image_dim=image_dim,
            window_center=window_center_ls,
            window_dim=window_dim
            )
        partial_sum = calc_optimal_centers.v_get_topleft_bottomright_partialsum(
            cumsum=cumsum,
            topleft=tl_ls,
            bottomright=br_ls,
        )
        if len(partial_sum) == 1:
            best_center = tl
            fraction_of_non_zero_pixels = partial_sum[0] / window_area
        else:
            best_sum = partial_sum.max()
            best_center_ls = window_center_ls[partial_sum == best_sum]
            if len(best_center_ls) == 1:
                best_center = best_center_ls[0]
            else:
                best_indices = best_center_ls - com
                best_idx = np.argmin((best_indices ** 2).sum(1))
                best_offset = best_indices[best_idx]
                best_center = com + best_offset
            fraction_of_non_zero_pixels = best_sum / window_area
        return {
            "window_dim_y": window_dim[0],
            "window_dim_x": window_dim[1],
            "best_center_y": best_center[0],
            "best_center_x": best_center[1],
            "fraction": fraction_of_non_zero_pixels,
        }
            
        
        
        centers = get_optimal_center_single(
                cropped_mammogram_path=fp,
                metadata=data
                )
        input()
        print(centers)


if __name__ == "__main__":
    main()
