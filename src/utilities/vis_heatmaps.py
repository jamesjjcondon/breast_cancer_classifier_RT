#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:06:10 2020

@author: mlim-user
"""
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from src.data_loading.drjc_datasets import BSSA_exams
from src.data_loading import loading
# from src.utilities.all_utils import show_2
from src.eval.helpers import add_result_columns
from src.constants import REPODIR, NVMEDIR, DATADIR, TTSDIR, VIEWS, ca_codes

from IPython import embed

	#%%

class vis_heatmaps():
    def __init__(self, parameters):
        self.parameters = parameters
        if parameters.infer_fp not in [None, False]:
            self.df = add_result_columns(
                    df=pd.read_csv(parameters.infer_fp),
                    VENDF_PATH=os.path.join(TTSDIR, 'dfs_with_vendors/balanced/test_df.csv'))

        self.image_extension = '.hdf5'
        train_trsfm = transforms.Compose([
#                    jc_transforms.CLAHE()
                    transforms.ToTensor(),
        #            transforms.Normalize((6780.65,), (8536.02,))
                ])
        
        self.ds = BSSA_exams(
                exam_list_in=parameters.data_path,
                transform=train_trsfm,
                parameters=self.parameters
                )
        self.ca_dicts = [x for x in self.ds.exam_list if 'e1' in x['L-CC'][0]]
        self.normies = [x for x in self.ds.exam_list if 'e0' in x['L-CC'][0]]
    
    def plot_one(self, ds_out, exam, view, save=False, dir=False, imdesc=None):
        if save:
            assert dir is not False
            assert imdesc is not None, "include a filename image descriptor"
        xs, ys = ds_out
        arr = xs[view]
        im, bhm, chm = [arr[i,:,:] for i in range(3)]
        _, attrs = loading.load_h5_image_and_attrs(
                        image_path=os.path.join(self.parameters.train_image_path, exam[view][0] + self.image_extension),
                        view=view,
                        horizontal_flip=exam["horizontal_flip"],
                    )
        if ys[view]['ca'] == 1:
            ca = ca_codes.get(attrs['HIST_SNOMED'])
        elif attrs['cancer_ep'] == 1:
            ca = 'no malignancy*'
        else:
            ca = 'no malignancy (>= 2yr f/up)'
            
        if ys[view]['benign'] == 1:
            ben = 'Bx-proven benign tumour'
        elif ys[view]['benign'] == 0 and attrs['cancer_ep'] == 1:
            ben = 'No benign tumour (Ca episode)'
        else:
            ben = 'no tumour (>= 2yr f/up)'
            
        title = 'NYU Pixel-level DenseNet121 (Wu et al) \ninference on BSSA'

        asp = .8 if view == 'MLO' else 16
        gs = {'width_ratios': [1,1,1,.1]} #, 'height_ratios':[1]}
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(8,6),
                                            subplot_kw={'aspect':asp},
                                            gridspec_kw=gs)
        fs = 18
        a1 = ax1.imshow(im, cmap='gray')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(view, fontdict={'fontsize':10})
                      #attrs['AN']
                      # , fontdict={'fontsize':fs})
    
        ax2.imshow(im, cmap='gray')
        a2 = ax2.imshow(bhm, cmap='jet', alpha=0.2) 
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title('Benign Heatmap', fontdict={'fontsize':10}) #:\n' + ben)
    
        ax3.imshow(im, cmap='gray') #, aspect='auto')
        a3 = ax3.imshow(chm, cmap='jet', alpha=0.2) #, aspect='auto')
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title('Malignant Heatmap \n' + ca, fontdict={'fontsize':10}) #, y=1.1)
        
        cax = ax4 #divider.append_axes("right", size="5%", pad=.1)
        frac = .5 if view == 'MLO' else .2
        cbar = fig.colorbar(a3, cax=cax, fraction=frac, pad=0.04)
        cbar.set_label("Probability of segmentation")

        top = .8 #5 if 'CC' in view else .8
        fig.subplots_adjust(wspace=None, hspace=top, top=top)
        plt.tight_layout()
        gs = gridspec.GridSpec(4, 1)
        ws = -.52 if 'CC' in view else -8

        gs.update(wspace=ws, hspace=0)
        if save:

            sfig = plt.gcf()
            filename = os.path.join(dir, attrs['AN'] + '_' + imdesc + '_' + view + '.jpg')
            sfig.savefig(filename, dpi=600, optimize=True, quality=95)
            print('\nsaved ', filename)
        else:
            plt.show()  

    def vis_one_center(self, title, image, exam, view, image_index):
        print(exam)
        print(exam['best_center'][view])
        y, x = exam['best_center'][view][image_index]
        plt.figure()
        plt.title(title)
        plt.imshow(image, cmap='gray')
        plt.scatter(x, y, s=50, c='red')
        plt.show()
        
    def vis_centers(self, n=2, AN=False, crops=False):
        
        random_number_generator = np.random.RandomState(22117)
        
        for i in range(n):
            self.exam = random.choice(self.ds.exam_list)
            if AN is not False:
                self.exam = [exam for exam in self.ds.exam_list if 'A'+str(AN) in exam['L-CC'][0]][0]

            loaded_image_dict = {view: [] for view in VIEWS.LIST}
            for view in VIEWS.LIST:
                image_index = random_number_generator.randint(low=0, high=len(self.exam[view]))

                s_image_name = self.exam[view][image_index]
                image_name = os.path.join(DATADIR, 'test_ims_master/cropped_images_sf2/'+s_image_name+'.hdf5')
                
                loaded_image = loading.load_image(image_name, view, horizontal_flip=self.exam['horizontal_flip'])
                loaded_image_dict[view].append(loaded_image)
                image = loaded_image
                self.vis_one_center(title=str(s_image_name),
                                    image=image, exam=self.exam, view=view,
                                    image_index=image_index)

                if crops:
                    for _ in range(5):
                        cropped_image, _ = loading.augment_and_normalize_image(
                                    image=loaded_image_dict[view][0],
                                    auxiliary_image=None, #loaded_heatmaps_dict[view][image_index],
                                    view=view,
                                    best_center=self.exam["best_center"][view][image_index],
                                    random_number_generator=random_number_generator,
                                    augmentation=True, #parameters["augmentation"],
                                    max_crop_noise=(100,100), #parameters["max_crop_noise"],
                                    max_crop_size_noise=100 #parameters["max_crop_size_noise"],
                                )
                        plt.figure()
                        plt.title('Crop'+view)
                        plt.imshow(cropped_image, cmap='gray')
                        plt.show()
                    
            if AN is not False:
                print('finishing AN {} images'.format(AN))
                break
                
    def show(self, arg, out_dir, save=False, benign_cols=False):
        if arg == 'ca':
            exam = random.choice(self.ca_dicts)
        elif arg == 'no_ca':
            exam = random.choice(self.normies)
        elif isinstance(arg, int):
            self.AN = 'A' + str(arg)
            try:
                exam = [exam for exam in self.ds.exam_list if self.AN in exam['L-CC'][0]][0]
            except IndexError as e:
                print('\n', e, '\n')
                print('\n', 
                      self.parameters.train_image_path, ' and ',
                      self.parameters.train_heatmaps_path,
                      ' might be empty???')
        ix = np.where(self.ds.exam_list == exam)[0][0]
        ds_out = self.ds.__getitem__(ix)
        if self.parameters.infer_fp not in [None, False]:
            pt = inst.df[inst.df.AN == self.AN]
            
            print('\nAccessionNumber:', self.AN, ' model ouput:')
            print('\nRight sided malignancy score (max from both views):')
            print(pt['right_malignant'].values)
            print('Left sided malignancy score (max from both views):')
            print(pt['left_malignant'].values)
            for col in ['left_malignant_MLO_MCPs', 'right_malignant_MLO_MCPs',
                        'left_malignant_CC_MCPs', 'right_malignant_CC_MCPs']:
                print('Model max class probability prediction ({}):'.format(col))
                print(pt[col].values)
                
            if benign_cols:
                print('\nRight sided benign score (max from both views):')
                print(pt['right_benign'].values)
                print('Left sided benign score (max from both views):')
                print(pt['left_benign'].values)
                for col in ['left_benign_MLO_MCPs', 'right_benign_MLO_MCPs',
                            'left_benign_CC_MCPs', 'right_benign_CC_MCPs']:
                    print('Model max class probability prediction ({}):'.format(col))
                    print(pt[col].values)
                
            
        for view, col in zip(VIEWS.LIST, ['left_malignant_CC_MCPs', 'right_malignant_CC_MCPs',
                                          'left_malignant_MLO_MCPs', 'right_malignant_MLO_MCPs']):
#            print(view, col)
#            print('\nAccessionNumber:', self.AN, ' model ouput:')
#            if 'R' in view:
#                
#            elif 'L' in view:
#                print('\nLeft sided malignancy score:')
#                print('\n', pt['left_malignant'].values)
#                
#            print('\nModel max class probability prediction ({}):'.format(col))
#            print(pt[col].values)
            print('\nImages/ Model input:')
                    
            self.plot_one(ds_out, exam, view,
                          save=save,
                          dir=out_dir,
                          imdesc='med')
            print('\n*************************************************')
            
          #%%  
if __name__== "__main__":
    
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--augmentation', type=str, default=None)

    parser.add_argument('--data-path', type=str, 
                        default=os.path.join(DATADIR, 'test_ims_master/NY_16-8pc_incidence/small_matched/data_NY16_sf_small_NYUC_matched.pkl'))
    parser.add_argument('--infer_fp', type=str,
                        default=os.path.join(
                            DATADIR,'test_ims_master/preds/small_matched_3c_NYU_init_preds.csv')
                        )
    parser.add_argument('--max_crop_noise', type=int, default=100)
    parser.add_argument('--max_crop_size_noise', type=int, default=100)

    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--train_heatmaps_path', type=str, 
                        default=os.path.join(DATADIR, 'test_ims_master/NY_16-8pc_incidence/small_matched/heatmaps_NY16_sf_small_matched'))
    parser.add_argument('--train_image_path', type=str, 
                        default=os.path.join(DATADIR, 'test_ims_master/NY_16-8pc_incidence/small_matched/cropped_NY16_ims_NYU_small'))
    parser.add_argument('--use_heatmaps', action='store_true', default=True)
    parser.add_argument('--use_hdf5', action='store_true', default=True)
    parser.add_argument('--use_n_exams', default=False)

    params = parser.parse_args()
    
    inst = vis_heatmaps(params)
    # inst.vis_centers(n=1, crops=False)
    inst.show(2686118,
              out_dir='/home/james/mydev/nyukat2.0/figures_and_tables',
              save=True
              ) #'ca') #163015, benign_cols=True) #2569877, benign_cols=True) # eg for paper: 2686118) #'ca') 
#    inst.show()
#    inst.show('no_ca')
        
