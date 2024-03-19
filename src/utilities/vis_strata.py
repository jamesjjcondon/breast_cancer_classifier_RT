#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 12:46:29 2020

@author: mlim-user
"""
import os
import random
import pandas as pd
import pickling
import matplotlib.pyplot as plt
from src.constants import TTSDIR
from src.utilities.all_utils import decode_col, load_df, revert_codes
from src.utilities.vis_exam import examine
from src.constants import VIEWS, NVMEDIR, DATADIR
from src.utilities.reading_images import read_image_mat

class vis_stratum:
    
    def __init__(self, cohort, split, dir, stratum, figsize):
        assert cohort in ['cases', 'controls', 'both']
        splits = ['train', 'val', 'test', 'all']
        assert split in splits
        old_cats = ['Stellate', 'Calcification', 'Discrete Mass with/without calcification',
       'Non-Specific Density', 'Architectural Distortion', 'Multiple Masses',
       'Other']
        new_cats = ["Stellate' or 'Multiple Masses'", 'Calcification',
       'Discrete Mass with/without calcification',
       "'Non-Specific Density', 'Architectural Distortion' or 'Other'"]
        if stratum not in [None, False]:
            assert stratum in old_cats or stratum in new_cats, print('possible strata are {} \nand: {}'.format(new_cats, old_cats))
        if cohort == 'both':
            print('\n\tloading both cases and controls for {} set'.format(split))
            name_cas, name_con = ['_'.join([split, cohort, 'df.csv']) for cohort in ['cases', 'controls']]
            df_cas = pd.read_csv(os.path.join(TTSDIR, name_cas))
            df_con = pd.read_csv(os.path.join(TTSDIR, name_con))
            self.df = pd.concat([df_cas, df_con], sort=False)
        else:
            name = '_'.join([split, cohort, 'df.csv'])
            self.df = pd.read_csv(os.path.join(TTSDIR, name))
        
        self.exam_list = pickling.unpickle_from_file(
                os.path.join(dir, split+'_ims_master/data_sf2.pkl'))
        self.cropped_im_dir = os.path.join(dir, split+'_ims_master/cropped_images_sf2')
        self.filenames = [next(iter(x[view])) for x in self.exam_list for view in VIEWS.LIST]
        self.cohort = cohort
        self.split = split
            
        _, self.codes = load_df(coded=True)
        self.stratum = stratum
        self.figsize= figsize
    
        snomed_codes = decode_col(self.codes, 'HIST_SNOMED.string()')
        self.df['HIST_SNOM_DECODE'] = self.df['HIST_SNOMED.string()'].apply(lambda x: snomed_codes[x])
        
        side_codes = decode_col(self.codes, 'AX_MAMM_SIDE_LESA_EV1.string()')
        self.df['AX_SIDE'] = self.df['AX_MAMM_SIDE_LESA_EV1.string()'].apply(lambda x: side_codes[x])
        
        strat_col = decode_col(self.codes, 'AX_WU_DOM_CAT_LESA_EV1.string()')
        self.df['AX_WU_DOM_CAT'] = self.df['AX_WU_DOM_CAT_LESA_EV1.string()'].apply(lambda x: strat_col[x])
        
        if stratum not in [None, False]:
            if self.stratum in new_cats:
                self.sdf = self.df[self.df['AX_WU_DOM_CAT'] == stratum]
            
            elif self.stratum in old_cats:
                self.df = revert_codes(self.df, col='AX_WU_DOM_CAT_LESA_EV1.string()')
                self.sdf = self.df[
                        self.df['AX_WU_DOM_CAT_LESA_EV1.string()_old'] == stratum
                        ]
            self.ANs = list(self.sdf['AN'].values)
        else:
            self.ANs = list(self.df['AN'].values)
    
    def slice_h5s(self, AN=None):
        if AN in [None, False]:
            AN = random.sample(self.ANs, 1)[0]
        else:
            AN = 'A'+str(AN)
        self.h5s = [x for x in self.filenames if AN in x]
        if len(self.h5s) < 4 :
            print('\nLess than 4 ims for {}. Randomly choosing another...')
            AN = random.sample(self.ANs, 1)[0]
        self.AN = AN

    def order_ims(self):
        self.exam = {
                'L-CC': read_image_mat(
                        os.path.join(
                                self.cropped_im_dir, 
                                [x for x in self.h5s if 'L-CC' in x][0]+'.hdf5'
                                )
                        ),
    
                'L-MLO': read_image_mat(
                        os.path.join(
                                self.cropped_im_dir, 
                                [x for x in self.h5s if 'L-MLO' in x][0]+'.hdf5'
                                )
                        ),
                'R-CC': read_image_mat(
                        os.path.join(
                                self.cropped_im_dir, 
                                [x for x in self.h5s if 'R-CC' in x][0]+'.hdf5'
                                )
                        ),
                'R-MLO': read_image_mat(
                        os.path.join(
                                self.cropped_im_dir, 
                                [x for x in self.h5s if 'R-MLO' in x][0]+'.hdf5'
                                )
                        ),
                }
        
    def plot(self):
        fig, ax = plt.subplots(2,2, figsize=self.figsize, squeeze=True) #, sharex=True, sharey=True)
        ax[0,1].imshow(self.exam['L-MLO'], cmap='gray')
        ax[0,1].set_xticklabels([])
        ax[0,1].set_yticklabels([])
        ax[0,0].imshow(self.exam['R-MLO'], cmap='gray')
        ax[0,0].set_xticklabels([])
        ax[0,0].set_yticklabels([])
        ax[1,1].imshow(self.exam['L-CC'], cmap='gray')
        ax[1,1].set_xticklabels([])
        ax[1,1].set_yticklabels([])
        
        ax[1,0].imshow(self.exam['R-CC'], cmap='gray')
        ax[1,0].set_xticklabels([])
        ax[1,0].set_yticklabels([])
        
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
#        plt.set_yticklabels([])
#        plt.set_xticklabels([])
        title = self.AN #+ '\nims for this pt: '+str(self.im_count)
        fig.suptitle(title)
        fig.subplots_adjust(wspace=0, hspace=0) #fig.squeeze() #tight_layout()
        fig.tight_layout()
        plt.show()
        
    def pt_data(self):
        self.pdf = self.df[self.df['AN'] == self.AN]
        
    def view_one(self):
        self.slice_h5s()
        self.order_ims()
        self.pt_data()
        print(self.stratum)
        print('\n Accession Number:', self.AN)
        print('SNOMED_histology', self.pdf['HIST_SNOM_DECODE'].values[0])
        print('SIDE (assessment, Lesion A):', self.pdf['AX_SIDE'].values[0])
        self.plot()
#%%
        
inst = vis_stratum('cases', 'test', dir=DATADIR,
                   stratum='Calcification', 
                   figsize=(20,20))

for i in range(10):
    print(i+1)
    inst.view_one()




#inst = examine(AN, 
#        folder='/home/mlim-user/Downloads/stelms')
#
#inst.AN_to_is_dir()
#inst.download()
#inst.arrs_views_lat()
#inst.choose_dups()
#inst.plot()