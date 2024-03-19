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
Runs the image only model and image+heatmaps model for breast cancer prediction.
"""  
import argparse
import collections as col
import numpy as np
import os
import pandas as pd
import torch
import tqdm

from torchvision import transforms

import src.utilities.pickling as pickling
import src.utilities.tools as tools
#from src.utilities.all_utils import show_heatmaps
import src.modeling.models as models
from src.modeling.pl_models import SplitBreastModel
from src.data_loading import loading, jc_transforms
from src.constants import DATADIR, NVMEDIR, VIEWS, VIEWANGLES, LABELS, MODELMODES

import matplotlib.pyplot as plt
#%%
def load_model(parameters):
    """
    Loads trained cancer classifier
    # modes are 
        - MODELMODES.IMAGE/'image'/src.modeling.models.ImageBreastModel/ all 4 views
    or (the default):
        - MODELMODES.VIEW_SPLIT/'view_split'/src.modeling.models.SplitBreastModel/ two views (CC, MLO) - default
    model.load_state_dict includes kwarg strict
    because keys in pytorch_lightning state_dict now include eg "epoch", "global_step" etc.
    """
    input_channels = 3 if parameters["use_heatmaps"] else 1
    model_class = {
        MODELMODES.VIEW_SPLIT: models.SplitBreastModel,
        MODELMODES.IMAGE: models.ImageBreastModel,
    }[parameters["model_mode"]]
    print('\n\tmodel_class', model_class)
    model = model_class(input_channels)
    print('\n\tLoading model {}...'.format(model))

    if parameters['retrained']:
        print('\n\tretrained version from', parameters['model_path'])
        model = SplitBreastModel.load_from_checkpoint(parameters['model_path'])
        
    else:
        print('\n\tnyukat / Wu version from', parameters['model_path'])
        model.load_state_dict(torch.load(parameters["model_path"])["model"])

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


def run_model(model, device, exam_list, parameters):
    """
    Returns predictions of image only model or image+heatmaps model.
    Prediction for each exam is averaged for a given number of epochs.
    """
    random_number_generator = np.random.RandomState(parameters["seed"])
    
    flips = [x['horizontal_flip'] for x in exam_list]
    if 'YES' in flips:
        print('some images are flipped per exam_list')
    image_extension = ".hdf5" if parameters["use_hdf5"] else ".png"
    if parameters["use_n_exams"] != -1:
        exam_list = exam_list[:parameters["use_n_exams"]]
        
    if parameters["extra_aug"]:
        print('\nUsing additional augmentations \n (flips and rotation)')
    
    print('\n\tlen exam_list:', len(exam_list))
    print('\n\tstarting inference loop...')
    with torch.no_grad():
        predictions_ls = []
        for datum in tqdm.tqdm(exam_list): # datum is 4-view examination
            BSSA_ID_pref = datum['L-CC'][0].rsplit('-', 3)[0] # take start of one filename (eg 'I777888-E2-A102778-S-i6') for logging 
            print(BSSA_ID_pref)
            predictions_for_datum = [] # for all 4-views
            loaded_image_dict = {view: [] for view in VIEWS.LIST}
            loaded_heatmaps_dict = {view: [] for view in VIEWS.LIST}
            for view in VIEWS.LIST:
                for short_file_path in datum[view]:
                    loaded_image = loading.load_image(
                        image_path=os.path.join(parameters["image_path"], short_file_path + image_extension),
                        view=view,
                        horizontal_flip=datum["horizontal_flip"],
                    )
                    if parameters["use_heatmaps"]:
                        loaded_heatmaps = loading.load_heatmaps(
                            benign_heatmap_path=os.path.join(parameters["heatmaps_path"], "heatmap_benign",
                                                             short_file_path + ".hdf5"),
                            malignant_heatmap_path=os.path.join(parameters["heatmaps_path"], "heatmap_malignant",
                                                                short_file_path + ".hdf5"),
                            view=view,
                            horizontal_flip=datum["horizontal_flip"],
                        )
                    else:
                        loaded_heatmaps = None

                    loaded_image_dict[view].append(loaded_image)
                    loaded_heatmaps_dict[view].append(loaded_heatmaps)
                    
            for data_batch in tools.partition_batch(range(parameters["num_epochs"]), parameters["batch_size"]):
                batch_dict = {view: [] for view in VIEWS.LIST}
                # batch_dict = {'L-CC': [], 'R-CC': [], 'L-MLO': [], 'R-MLO': []}
                for _ in data_batch: # data_batch is range(batch_size)
                    for view in VIEWS.LIST:
                        #print('n', view)
                        image_index = 0
                        if parameters["augmentation"]:
                            image_index = random_number_generator.randint(low=0, high=len(datum[view]))
                        cropped_image, cropped_heatmaps = loading.augment_and_normalize_image(
                            image=loaded_image_dict[view][image_index],
                            auxiliary_image=loaded_heatmaps_dict[view][image_index],
                            view=view,
                            best_center=datum["best_center"][view][image_index],
                            random_number_generator=random_number_generator,
                            augmentation=parameters["augmentation"],
                            max_crop_noise=parameters["max_crop_noise"],
                            max_crop_size_noise=parameters["max_crop_size_noise"],
                        )
                        
#                        print(type(cropped_image))
#                        print(type(cropped_heatmaps))
#                        print(cropped_heatmaps.shape)
#                        print(cropped_heatmaps)
                        
#                        show_heatmaps(
#                                title=BSSA_ID_pref, im=cropped_image,
#                                bhm=cropped_heatmaps[:,:,0],
#                                chm=cropped_heatmaps[:,:,1],
#                                view=view, AN='None',
#                                yben='None', yca='None')
                        if loaded_heatmaps_dict[view][image_index] is None:
                            model_im = cropped_image[:, :, np.newaxis]
#                            batch_dict[view].append(cropped_image[:, :, np.newaxis])
                        else:
                            model_im = np.concatenate([
                                cropped_image[:, :, np.newaxis],
                                cropped_heatmaps,
                            ], axis=2)
    
#                        pickling.pickle_to_file(
#                                '/home/james/Downloads/test_im',
#                                model_im
#                                )
                        if parameters["extra_aug"]:
                            all_tfs = transforms.Compose(
                                    [jc_transforms.RandomHorizontalFlip(p=0.5),
                                     jc_transforms.RandomVerticalFlip(p=0.5),
                                     jc_transforms.RandomRotate(p=0.5, rrange=(0,360))
                                     ])
                            model_im = all_tfs(model_im)
                            
#                        plt.figure(figsize=(10,10))
#                        plt.imshow(model_im[:,:,0], cmap='gray')
#                        plt.show()
#                        input()
                        
                        batch_dict[view].append(model_im)

                tensor_batch = { # send dict of for views to GPU
                    view: torch.tensor(np.stack(batch_dict[view])).permute(0, 3, 1, 2).to(device)
                    for view in VIEWS.LIST
                }
                
                
                output = model(tensor_batch)
#                print(output)
#                print(output[view].shape() for view in VIEWS.LIST)
#                input()
#                if parameters['retrained']: # with logits
#                    for key in output.keys():
#                        output[key] = F.softmax(output[key], dim=-1)
                batch_predictions = compute_batch_predictions(output, parameters)
                pred_df = pd.DataFrame({k: v[:, 1] for k, v in batch_predictions.items()})
                pred_df.columns.names = ["label", "view_angle"]
                predictions = pred_df.T.reset_index().groupby("label").mean().T[LABELS.LIST].values
#                print(predictions.shape)
#                print(type(predictions))
#                print('\tpredictions:', predictions)
#                predictions['BSSA_ID'] = BSSA_ID_pref
                predictions_for_datum.append(predictions)
#                print('\n\tpredictions_for_datum:', predictions_for_datum)
            predictions_ls.append([BSSA_ID_pref, np.mean(np.concatenate(predictions_for_datum, axis=0), axis=0)])
            ls_path = os.path.split(parameters['output_path'])[0]
            if not os.path.exists(ls_path):
                os.mkdir(ls_path)
            pickling.pickle_to_file(
                    os.path.join(ls_path, 'running_pred_ls.pkl'),
                    predictions_ls
                    )
#            print('\n\npredictions_ls:', predictions_ls)
#            print(type(predictions_ls))
    return predictions_ls


def compute_batch_predictions(y_hat, parameters):
    """
    Format predictions from different heads
    """
    
    if parameters["model_mode"] == MODELMODES.VIEW_SPLIT:
        assert y_hat[VIEWANGLES.CC].shape[1:] == (4, 2)
        assert y_hat[VIEWANGLES.MLO].shape[1:] == (4, 2)
        batch_prediction_tensor_dict = col.OrderedDict()
        batch_prediction_tensor_dict[LABELS.LEFT_BENIGN, VIEWANGLES.CC] = y_hat[VIEWANGLES.CC][:, 0]
        batch_prediction_tensor_dict[LABELS.LEFT_BENIGN, VIEWANGLES.MLO] = y_hat[VIEWANGLES.MLO][:, 0]
        batch_prediction_tensor_dict[LABELS.RIGHT_BENIGN, VIEWANGLES.CC] = y_hat[VIEWANGLES.CC][:, 1]
        batch_prediction_tensor_dict[LABELS.RIGHT_BENIGN, VIEWANGLES.MLO] = y_hat[VIEWANGLES.MLO][:, 1]
        batch_prediction_tensor_dict[LABELS.LEFT_MALIGNANT, VIEWANGLES.CC] = y_hat[VIEWANGLES.CC][:, 2]
        batch_prediction_tensor_dict[LABELS.LEFT_MALIGNANT, VIEWANGLES.MLO] = y_hat[VIEWANGLES.MLO][:, 2]
        batch_prediction_tensor_dict[LABELS.RIGHT_MALIGNANT, VIEWANGLES.CC] = y_hat[VIEWANGLES.CC][:, 3]
        batch_prediction_tensor_dict[LABELS.RIGHT_MALIGNANT, VIEWANGLES.MLO] = y_hat[VIEWANGLES.MLO][:, 3]
#        if parameters['retrained']:
#            batch_prediction_dict = col.OrderedDict([
#            (k, v.cpu().detach().numpy()) #EXPENSIVE
#            for k, v in batch_prediction_tensor_dict.items()
#        ])
#        else:
        batch_prediction_dict = col.OrderedDict([
        (k, np.exp(v.cpu().detach().numpy())) #EXPENSIVE
        for k, v in batch_prediction_tensor_dict.items()
        ])
    elif parameters["model_mode"] == MODELMODES.IMAGE:
        assert y_hat[VIEWS.L_CC].shape[1:] == (2, 2)
        assert y_hat[VIEWS.R_CC].shape[1:] == (2, 2)
        assert y_hat[VIEWS.L_MLO].shape[1:] == (2, 2)
        assert y_hat[VIEWS.R_MLO].shape[1:] == (2, 2)
        batch_prediction_tensor_dict = col.OrderedDict()
        batch_prediction_tensor_dict[LABELS.LEFT_BENIGN, VIEWS.L_CC] = y_hat[VIEWS.L_CC][:, 0]
        batch_prediction_tensor_dict[LABELS.LEFT_BENIGN, VIEWS.L_MLO] = y_hat[VIEWS.L_MLO][:, 0]
        batch_prediction_tensor_dict[LABELS.RIGHT_BENIGN, VIEWS.R_CC] = y_hat[VIEWS.R_CC][:, 0]
        batch_prediction_tensor_dict[LABELS.RIGHT_BENIGN, VIEWS.R_MLO] = y_hat[VIEWS.R_MLO][:, 0]
        batch_prediction_tensor_dict[LABELS.LEFT_MALIGNANT, VIEWS.L_CC] = y_hat[VIEWS.L_CC][:, 1]
        batch_prediction_tensor_dict[LABELS.LEFT_MALIGNANT, VIEWS.L_MLO] = y_hat[VIEWS.L_MLO][:, 1]
        batch_prediction_tensor_dict[LABELS.RIGHT_MALIGNANT, VIEWS.R_CC] = y_hat[VIEWS.R_CC][:, 1]
        batch_prediction_tensor_dict[LABELS.RIGHT_MALIGNANT, VIEWS.R_MLO] = y_hat[VIEWS.R_MLO][:, 1]

        batch_prediction_dict = col.OrderedDict([
            (k, np.exp(v.cpu().detach().numpy())) # EXPENSIVE
            for k, v in batch_prediction_tensor_dict.items()
        ])
    else:
        raise KeyError(parameters["model_mode"])
    return batch_prediction_dict


def load_run_save(data_path, output_path, parameters):
    """
    Outputs the predictions as csv file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    assert output_path.endswith('.csv')
    exam_list = pickling.unpickle_from_file(data_path)
    model, device = load_model(parameters)
    predictions = run_model(model, device, exam_list, parameters)
    # Take the positive prediction
#    print('\n\npredictions:')
#    print(predictions)
#    print(type(predictions))
    preds = [x[1] for x in predictions]
    IDs = [x[0] for x in predictions]
#    input()
#    print('\n\nIDs:', IDs)
#    print(type(IDs))
    df = pd.DataFrame(preds, columns=LABELS.LIST)
    df['BSSA_ID'] = IDs
    df.to_csv(output_path, index=False, float_format='%.4f')

#%%
def main():
    parser = argparse.ArgumentParser(description='Run image-only model or image+heatmap model')
    parser.add_argument('--model-mode', default=MODELMODES.VIEW_SPLIT, type=str)
    parser.add_argument('--model-path',
#                        default='/home/mlim-user/Documents/james/NYU_retrain_SSD/train_models/ckpts_21Mar_wd2e-06_v1/21Mar_wd2e-06_v1_epo=14_tloss=0.926_avloss=2.035.ckpt')
                        default=os.path.join(DATADIR,
                                             #'dev_test_models/ckpts_18Mar_dev_v0/18Mar_dev_v0_epo=54_tloss=1.102_avloss=1.533.ckpt'
                                             'train_models/ckpts_21Mar_wd2e-06_v1/21Mar_wd2e-06_v1_epo=14_tloss=0.926_avloss=2.035.ckpt'
                                             )
                        )
    parser.add_argument('--data-path',
#                        default='/home/mlim-user/Documents/james/NYU_retrain_SSD/val_ims_sf2/data_sf2.pkl')
                        default=os.path.join(NVMEDIR, 'test_ims_master/data_sf2.pkl'))
    parser.add_argument('--image-path',
                        default=os.path.join(NVMEDIR, 'test_ims_master/cropped_images_sf2'))
    parser.add_argument('--output-path',
                        default=os.path.join(NVMEDIR,
                                             #'dev_test_models/ckpts_18Mar_dev_v0/50-50-preds/imageheatmap_predictions.csv'
                                             #'train_models/ckpts_21Mar_wd2e-06_v1/preds_5050_21Mar_wd2e-06v1/redo1_val_test/imageheatmaps_predictions.csv'
                                             'test_ims_master/5050_preds_sf2/Fine-tuned-xaug/imageheatmaps_predictions.csv'
                                             )
                        )
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--seed', default=22117, type=int)
    parser.add_argument('--use-heatmaps', action="store_true", default=True)
    parser.add_argument('--heatmaps-path',
                        default=os.path.join(NVMEDIR, 'test_ims_master/heatmaps_sf2'))
    parser.add_argument('--use-augmentation', action="store_true", default=True)
    parser.add_argument('--extra-aug', action="store_true", default=True)
    parser.add_argument('--use-hdf5', action="store_true", default=True)
    parser.add_argument('--num-epochs', default=10, type=int)
    parser.add_argument('--device-type', default="gpu", choices=['gpu', 'cpu'])
    parser.add_argument('--gpu-number', type=int, default=0)
    parser.add_argument('--retrained', action='store_true', default=True)
    parser.add_argument('--use-n-exams', default=-1)
    args = parser.parse_args()

    parameters = {
        "device_type": args.device_type,
        "gpu_number": args.gpu_number,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "image_path": args.image_path,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "augmentation": args.use_augmentation,
        "extra_aug": args.extra_aug,
        "num_epochs": args.num_epochs,
        "use_heatmaps": args.use_heatmaps,
        "heatmaps_path": args.heatmaps_path,
        "use_hdf5": args.use_hdf5,
        "model_mode": args.model_mode,
        "model_path": args.model_path,
        "output_path": args.output_path,
        "retrained": args.retrained,
        "use_n_exams": int(args.use_n_exams)
    }
    
#    parameters = {
#        "device_type": 'gpu',
#        "gpu_number": 0,
#        "max_crop_noise": (100, 100),
#        "max_crop_size_noise": 100,
#        "image_path": '/home/mlim-user/Documents/james/NYU_retrain_SSD/val_ims_sf2/cropped_images_sf2',
#        "batch_size": 1,
#        "seed": 1,
#        "augmentation": True,
#        "num_epochs": 10,
#        "use_heatmaps": True,
#        "heatmaps_path": '/home/mlim-user/Documents/james/NYU_retrain_SSD/val_ims_sf2/heatmaps_sf2',
#        "use_hdf5": True,
#        "model_mode": MODELMODES.VIEW_SPLIT,
#        "model_path": '/home/mlim-user/Documents/james/NYU_retrain_SSD/train_models/ckpts_19Mar_v1/19Mar_v2_epo=28_tloss=1.117_avloss=2.053.ckpt',
#            #'/media/james/drjc_ext_HD/data/NYU_retrain/train_models/ckpts_22Mar_log_sm_v1/22Mar_log_sm_v1_epo=12_tloss=2.549_avloss=2.553.ckpt',
#            #'/media/2TB_HDD/NYU_retrain/dev_test_models/ckpts_18Mar_dev_v0/18Mar_dev_v0_epo=54_tloss=1.102_avloss=1.533.ckpt',
#        "retrained": True,
#        "use_n_exams": 4
#    }
#    data_path = '/home/mlim-user/Documents/james/NYU_retrain_SSD/val_ims_sf2/data_sf2.pkl'
#    
    load_run_save(
        data_path=args.data_path,
        output_path=args.output_path,
        parameters=parameters,
    )


if __name__ == "__main__":
    main()
