#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:18:30 2020

loads model-ready nyu_style dictionary, 
correlates with BSSA data
prints counts

@author: mlim-user
"""
import os
import numpy as np
import pandas as pd
import argparse
import itertools
from src.utilities import pickling
from src.utilities.all_utils import load_df, decode_col, data_to_df
from src.constants import TTSDIR, DATADIR, NVMEDIR, VIEWS

splits = ['train', 'val', 'test']
cohorts = ['cases', 'controls']

folders = ['train_ims_master', 'val_ims_master', 'test_ims_master']

#%%

class explore_cohort:
    def __init__(self, cohort, split, incidence, exam_list_fp, dir):
        assert cohort in ['cases', 'controls', 'both']
        splits = ['train', 'val', 'test', 'all']
        assert split in splits
        self.cropped_im_dir = os.path.join(dir, split+'_ims_master/cropped_images_sf2')
        self.exam_list = pickling.unpickle_from_file(exam_list_fp)

        if cohort == 'both':
            assert incidence not in [None, False]
            print('\n\tloading both cases and controls for {} set'.format(split))
            self.df = data_to_df('/data/james/NYU_retrain/test_ims_master/data_sf2.pkl')
            
        elif incidence == 16.8:
            self.df = data_to_df(self.exam_list)
            if cohort == 'controls':
                self.df = self.df[(self.df['cancer_ep'] == 0) & (self.df['benign_pt'] == 0)]
            elif cohort == 'cases':
                self.df = self.df[(self.df['cancer_ep'] == 1) or (self.df['benign_pt'] == 1)]
            
            self.cropped_im_dir = os.path.join(dir, split+'_ims_master/NY_16-8pc_incidence/cropped_images_sf2')

        else:
            name = '_'.join([split, cohort, 'df.csv'])
            self.df = pd.read_csv(os.path.join(TTSDIR, name))
            
        self.filenames = [next(iter(x[view])) for x in self.exam_list for view in VIEWS.LIST]
        self.cohort = cohort
        self.split = split
        self.dir = dir
            
        _, self.codes = load_df(coded=True)
        snomed_codes = decode_col(self.codes, 'HIST_SNOMED.string()')
        self.df['HIST_SNOM_DECODE'] = self.df['HIST_SNOMED.string()'].apply(lambda x: snomed_codes[x])
    
    def compare_ttsdf_and_examlist(self):
        print('\nAsserting all df ANs are in exam_list...')
        self.ANs = []
        self.not_found = []
        for AN in self.df.AN.values: # for accession numbers in df
            if any(AN in filename for filename in self.filenames): # if AN is found in any filenames...
#                print('found df.AN {} in '.format(AN) + split + ' exam_list')
                self.ANs.append(AN)
                pass
            else:
                print('{} not_found'.format(AN) + ' in exam_list')
                self.not_found.append(AN)
    
    def update_dfs(self):
        print('\n df shape before removing ANs not in exam_list:')
        print(self.df.shape)
        self.df = self.df[self.df.AN.isin(self.ANs)]
        print('\n df shape after removing ANs not in exam_list:')
        print(self.df.shape)
        print('\nn ANs not found: ', len(self.not_found))
#        self.df = self.df[[res_cols]]
        
    def check_files_on_disk(self):
        self.files = sorted(os.listdir(self.cropped_im_dir))
        self.ondiskANs = [x.split('-',3)[2] for x in self.files]
        self.not_on_disk = 0
        for AN in self.df['AN']:
            if AN in self.ondiskANs:
                pass
            else:
                print("{} not on disk (".format(AN)+self.cropped_im_dir+')')
                self.not_on_disk += 1
        if self.not_on_disk == 0:
            print('\nAll patients have at least one hdf5 in', self.cropped_im_dir)
        else:
            print(self.not_on_disk, "\npatients' ANs not found in ", self.cropped_im_dir)
        
    def count_ims(self):
        self.all_ims = [x[view] for x in self.exam_list for view in VIEWS.LIST]
        self.all_ims = list(itertools.chain.from_iterable(self.all_ims))
        self.total_ims = len(self.all_ims)
        
        print('\n', str(self.total_ims) + ' images in ', self.split + ' ({})'.format(self.dir))
        self.cohort_ims = [x for x in self.all_ims if x.split('-',3)[-2] in self.ANs]
        print('\n\t{} unique patient IDs'.format(len(np.unique(self.df['ID.int32()']))))
        print('\n\t{} unique episodes (ANs)'.format(len(np.unique(self.ANs))))
        print('\n\t', len(self.cohort_ims), 'images for', self.cohort)
        print('\n\t{} malignancies'.format(sum(self.df['cancer_ep'])))
        print('\n\t{} benign tumours'.format(sum(self.df['benign_tumour_ep'])))

    def save_catcol_sum_df(self, col, newname):
        ''' Summary dataframe for one categorical column '''
        sdf = self.df[col].value_counts(dropna=False)
        sdf.name = 'n'
        pc = np.round(self.df[col].value_counts(dropna=False, normalize=True), 3)
        pc.name = '%'
        sdf = pd.concat([sdf, pc], axis=1).reset_index()
        colcodes = decode_col(self.codes, col)
        sdf[newname] = sdf['index'].apply(lambda x: colcodes[x])
        sdf.drop(columns='index', inplace=True)
        cols = sdf.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        self.sdf = sdf[cols].sort_values(newname)
        if col == 'HIST_SNOMED.string()' and self.cohort == 'both':
            self.sdf.loc[0, newname] = '(Control - never biopsied)'
        print('\n', sdf)
        return self.sdf

#%%
if __name__ == "__main__":
#    parser = argparse.ArgumentParser(add_help=False)
#    parser.add_argument('--exam_list_fp', action='store_true', default=True)
    split = 'train'
    cohort = 'cases'

    inst = explore_cohort(
            cohort=cohort,
            split=split,
            incidence=44, #16.8,
#            exam_list_fp = #'/data/james/test_ims_master/data_sf2.pkl',
            exam_list_fp = '/data/james/NYU_retrain/test_ims_master/NY_16-8pc_incidence/data_sf2.pkl',
            dir=DATADIR)

    inst.compare_ttsdf_and_examlist()
    inst.update_dfs()
    inst.check_files_on_disk()
    inst.count_ims()
#    for col in inst.df.columns:
#        inst.save_catcol_sum_df(col, col+'_new')
    # column = 'HIST_SNOMED.string()'
    # newcolname = 'val_SNOMED'
    # csvoutpath = '/home/mlim-user/Documents/james/my_dev/nyukat2.0/figures_and_tables/{}.csv'.format(newcolname)
    
#    df = inst.save_catcol_sum_df(column, newcolname)
#    df.to_csv(csvoutpath, index=False)
#    hparams = parser.parse_args()
    

            
    
