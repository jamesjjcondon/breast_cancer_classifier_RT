# Copyright (C) 2019 Nan Wu, Jason Phang, Jungkyu Park, Yiqiu Shen, Zhe Huang, Masha Zorin, 
#   Stanisław Jastrzębski, Thibault Févry, Joe Katsnelson, Eric Kim, Stacey Wolfson, Ujas Parikh, 
#   Sushma Gaddam, Leng Leng Young Lin, Kara Ho, Joshua D. Weinstein, Beatriu Reig, Yiming Gao, 
#   Hildegard Toth, Kristine Pysarenko, Alana Lewin, Jiyon Lee, Krystal Airola, Eralda Mema, 
#   Stephanie Chung, Esther Hwang, Naziya Samreen, S. Gene Kim, Laura Heacock, Linda Moy, 
#   Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of breast_cancer_classifier.
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
Generates benign and malignant heatmaps for cropped images using patch classifier. 
"""   
import numpy as np
import random
import os
import argparse
import tqdm

import torch
import torch.nn.functional as F

import src.heatmaps.models as models
import src.data_loading.loading as loading
import src.utilities.pickling as pickling
import src.utilities.saving_images as saving_images
import src.utilities.tools as tools
from src.constants import VIEWS

#%%

def stride_list_generator(img_width, patch_size, more_patches=0, stride_fixed=-1):
    """
    Determines how an image should be split up into patches 
    """
    if stride_fixed != -1:
        patch_num_lower_bound = (img_width - patch_size) // stride_fixed + 1
        pixel_left = (img_width - patch_size) % stride_fixed
        more_patches = 0
    else:
        patch_num_lower_bound = img_width // patch_size
        pixel_left = img_width % patch_size
        stride_fixed = patch_size
        
    if pixel_left == 0 and more_patches == 0:
        stride = stride_fixed
        patch_num = patch_num_lower_bound
        sliding_steps = patch_num - 1
        stride_list = [stride] * sliding_steps
    else:
        pixel_overlap = stride_fixed - pixel_left + more_patches * stride_fixed
        patch_num = patch_num_lower_bound + 1 + more_patches
        sliding_steps = patch_num - 1
        
        stride_avg = stride_fixed - pixel_overlap // sliding_steps
        
        sliding_steps_smaller = pixel_overlap % sliding_steps
        stride_smaller = stride_avg - 1
        
        stride_list = [stride_avg] * sliding_steps

        for step in random.sample(range(sliding_steps), sliding_steps_smaller):
            stride_list[step] = stride_smaller
            
    return stride_list


def prediction_by_batch(minibatch_patches, model, device, parameters):
    """
    Puts patches into a batch and gets predictions of patch classifier.
    """
    minibatch_x = np.stack((minibatch_patches,) * parameters['input_channels'], axis=-1).reshape(
        -1, parameters['patch_size'], parameters['patch_size'], parameters['input_channels']
    ).transpose(0, 3, 1, 2)

    with torch.no_grad():
        output = F.softmax(model(torch.FloatTensor(minibatch_x).to(device)), dim=1).cpu().detach().numpy()
    return output


def ori_image_prepare(image_path, view, horizontal_flip, parameters):
    """
    Loads an image and creates stride_lists
    """
    patch_size = parameters['patch_size']
    more_patches = parameters['more_patches']
    stride_fixed = parameters['stride_fixed']

    image = loading.load_image(image_path, view, horizontal_flip)
    image = image.astype(float)
    loading.standard_normalize_single_image(image)
    
    img_width, img_length = image.shape
    width_stride_list = stride_list_generator(img_width, patch_size, more_patches, stride_fixed)
    length_stride_list = stride_list_generator(img_length, patch_size, more_patches, stride_fixed)

    return image, width_stride_list, length_stride_list


def patch_batch_prepare(image, length_stride_list, width_stride_list, patch_size):
    """
    Samples patches from an image according to stride_lists
    """
    min_x, min_y = 0, 0
    minibatch_patches = []
    img_width, img_length = image.shape

    for stride_y in length_stride_list + [0]:
        for stride_x in width_stride_list + [-(img_width - patch_size)]:
            patch = image[min_x:min_x + patch_size, min_y:min_y + patch_size]
            minibatch_patches.append(np.expand_dims(patch, axis=2))
            min_x += stride_x
        min_y += stride_y
    
    return minibatch_patches


def probabilities_to_heatmap(patch_counter, all_prob, image_shape, length_stride_list, width_stride_list,
                             patch_size, heatmap_type):
    """
    Generates heatmaps using output of patch classifier
    """
    min_x, min_y = 0, 0
    
    prob_map = np.zeros(image_shape, dtype=np.float32)
    count_map = np.zeros(image_shape, dtype=np.float32)
    
    img_width, img_length = image_shape

    for stride_y in length_stride_list + [0]:
        for stride_x in width_stride_list + [-(img_width - patch_size)]:
            prob_map[min_x:min_x + patch_size, min_y:min_y + patch_size] += all_prob[
                patch_counter, heatmap_type
            ]
            count_map[min_x:min_x + patch_size, min_y:min_y + patch_size] += 1
            min_x += stride_x
            patch_counter += 1
        min_y += stride_y
    
    heatmap = prob_map / count_map
    
    return heatmap, patch_counter


def get_all_prob(all_patches, minibatch_size, model, device, parameters):   
    """
    Gets predictions for all sampled patches
    """
    all_prob = np.zeros((len(all_patches), parameters['number_of_classes']))

    for i, minibatch in enumerate(tools.partition_batch(all_patches, minibatch_size)):
        minibatch_prob = prediction_by_batch(minibatch, model, device, parameters)
        all_prob[i * minibatch_size: i * minibatch_size + minibatch_prob.shape[0]] = minibatch_prob
                
    return all_prob.astype(np.float32)


def save_heatmaps(heatmap_malignant, heatmap_benign, short_file_path, view, horizontal_flip, parameters):
    """
    Saves the heatmaps after flipping back to the original direction
    """
    heatmap_malignant = loading.flip_image(heatmap_malignant, view, horizontal_flip)
    heatmap_benign = loading.flip_image(heatmap_benign, view, horizontal_flip)
    heatmap_save_path_malignant = os.path.join(
        parameters['save_heatmap_path'][0], 
        short_file_path + '.hdf5'
    )
    saving_images.save_image_as_hdf5(heatmap_malignant, heatmap_save_path_malignant)

    heatmap_save_path_benign = os.path.join(
        parameters['save_heatmap_path'][1],
        short_file_path + '.hdf5'
    )
    saving_images.save_image_as_hdf5(heatmap_benign, heatmap_save_path_benign)

def check_heatmaps(short_file_path, map_fps, view, horizontal_flip):
    
    heatmap_save_path_malignant = os.path.join(map_fps[0], short_file_path + '.hdf5')
    heatmap_save_path_benign = os.path.join(map_fps[1], short_file_path + '.hdf5')

    heatmaps = loading.load_heatmaps(
            benign_heatmap_path=heatmap_save_path_benign, 
            malignant_heatmap_path=heatmap_save_path_malignant, 
            view=view, 
            horizontal_flip=horizontal_flip
            )
    assert heatmaps is not None
    
def get_image_path(short_file_path, parameters):
    """
    Convert short_file_path to full file path
    """
    image_extension = '.hdf5' if parameters['use_hdf5'] else '.png'
    return os.path.join(parameters['original_image_path'], short_file_path + image_extension)


def sample_patches(exam, parameters):
    """
    Samples patches for one exam
    """
    all_patches = []
    all_cases = []
    for view in VIEWS.LIST:
        for short_file_path in exam[view]:
            image_path = get_image_path(short_file_path, parameters)
            patches, case = sample_patches_single(
                image_path=image_path,
                view=view,
                horizontal_flip=exam['horizontal_flip'],
                parameters=parameters,
            )

            all_patches += patches
            all_cases.append([short_file_path] + case)

    return all_patches, all_cases


def sample_patches_single(image_path, view, horizontal_flip, parameters):
    """
    Sample patches for a single mammogram image
    """
    image, width_stride_list, length_stride_list = ori_image_prepare(
        image_path,
        view,
        horizontal_flip,
        parameters,
    )
    patches = patch_batch_prepare(
        image,
        length_stride_list,
        width_stride_list,
        parameters['patch_size'],
    )
    case = [
        image.shape,
        view,
        horizontal_flip,
        width_stride_list,
        length_stride_list,
    ]
    return patches, case


def making_heatmap_with_large_minibatch_potential(parameters, model, exam_list, device):
    """
    Samples patches for each exam, gets batch prediction, creates and saves heatmaps
    """
    minibatch_size = parameters['minibatch_size']
    
    if not os.path.exists(parameters['save_heatmap_path'][0]):
        os.makedirs(parameters['save_heatmap_path'][0]) #, exist_ok=True)
    if not os.path.exists(parameters['save_heatmap_path'][1]):
        os.makedirs(parameters['save_heatmap_path'][1]) #, exist_ok=True)
        
    print('\nLength of exam_list needing heatmaps:', len(exam_list))
    completed_maps_fp = os.path.join(parameters['done_maps_dic_fp'], 'completed_maps.pkl')
    if os.path.exists(completed_maps_fp):
        print('\n\tCompleted heatmaps found. Removing from to do list...')
        completed_list = pickling.unpickle_from_file(completed_maps_fp)
        for exam in completed_list:
            exam_list.remove(exam)
        print('\n {} heatmaps to do.'.format(len(exam_list)))
    else:
        completed_list = []
#    print('\nexisting_heatmaps')
#    print(existing_heatmaps)
    for exam in tqdm.tqdm(exam_list):
        
#        if any(exam['R-MLO'][0] in x for x in existing_heatmaps):
#            print(exam)
#            print(type(exam))
#            continue
        # create patches and other information with the images
        all_patches, all_cases = sample_patches(exam, parameters)
#        print(type(all_cases))

        if len(all_patches) != 0:
            all_prob = get_all_prob(
                all_patches, 
                minibatch_size, 
                model,
                device,
                parameters
            )
        
            del all_patches
            
            patch_counter = 0
        
            for (short_file_path, image_shape, view, horizontal_flip, width_stride_list, length_stride_list) \
                    in all_cases:

                heatmap_malignant, _ = probabilities_to_heatmap(
                    patch_counter, 
                    all_prob, 
                    image_shape, 
                    length_stride_list, 
                    width_stride_list, 
                    parameters['patch_size'], 
                    parameters['heatmap_type'][0]
                )
                heatmap_benign, patch_counter = probabilities_to_heatmap(
                    patch_counter, 
                    all_prob, 
                    image_shape, 
                    length_stride_list, 
                    width_stride_list, 
                    parameters['patch_size'], 
                    parameters['heatmap_type'][1]
                )
                save_heatmaps(
                    heatmap_malignant, 
                    heatmap_benign, 
                    short_file_path, 
                    view, 
                    horizontal_flip, 
                    parameters
                )

                del heatmap_malignant, heatmap_benign
                
            del all_prob, all_cases
            
        completed_list.append(exam)
        pickling.pickle_to_file(
                file_name=completed_maps_fp,
                data=completed_list
                )


def load_model(parameters):
    """
    Load trained patch classifier
    """
    model = models.ModifiedDenseNet121(num_classes=parameters['number_of_classes'])
    model.load_from_path(parameters["initial_parameters"])
    
    if torch.cuda.device_count() > 1:
        if parameters["gpu_number"] != -1:
            print('\n\tUsing gpu num', parameters["gpu_number"])
            device = torch.device("cuda:{}".format(parameters["gpu_number"]))
        else:
            print('\n\tUsing torch.nn.DataParallel')
            device = torch.device('cuda')
            model = torch.nn.DataParallel(model)
    
    else:        
        if (parameters["device_type"] == "gpu") and torch.has_cudnn:
            device = torch.device("cuda:{}".format(parameters["gpu_number"]))
        else:
            device = torch.device("cpu")
    
    model = model.to(device)
    model.eval()
    return model, device


def produce_heatmaps(model, device, parameters):
    """
    Generates heatmaps for all exams
    """
    # Load exam info
    exam_list = pickling.unpickle_from_file(parameters['data_file'])    

    # Create heatmaps
    making_heatmap_with_large_minibatch_potential(parameters, model, exam_list, device)


def main():
    parser = argparse.ArgumentParser(description='Produce Heatmaps')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--image-path', required=True)
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--output-heatmap-path', required=True)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
    parser.add_argument("--gpu-number", type=int, default=0)
    parser.add_argument("--use-hdf5", action="store_true")
    parser.add_argument("--shrink-factor", default=None)
    args = parser.parse_args()

    
    
    #if args.shrink_factor:
#    if args.shrink_factor == 'match':
#        args.shrink_factor = 1.9 # mean of 'CC' and 'MLO' (different sizes at BSSA)
#        #patch_size = int(256/args.shrink_factor)
        
    #else:
        #minibatch_size = 500
        #patch_size = 256
        
    parameters = dict(
        device_type=args.device_type,
        gpu_number=args.gpu_number,
        
        patch_size=256,

        stride_fixed=70,
        more_patches=5,
        minibatch_size=args.batch_size,
        seed=args.seed,
        
        initial_parameters=args.model_path,
        input_channels=3,
        number_of_classes=4,
        
        data_file=args.data_path,
        original_image_path=args.image_path,
        save_heatmap_path=[os.path.join(args.output_heatmap_path, 'heatmap_malignant'),
                           os.path.join(args.output_heatmap_path, 'heatmap_benign')],
        done_maps_dic_fp=args.output_heatmap_path,
        heatmap_type=[0, 1],  # 0: malignant 1: benign 0: nothing

        use_hdf5=args.use_hdf5
    )
    print('\npatch_size:', parameters['patch_size'])
    random.seed(parameters['seed'])
    model, device = load_model(parameters)
    produce_heatmaps(model, device, parameters)


if __name__ == "__main__":
    main()
