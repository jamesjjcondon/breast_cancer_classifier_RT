#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:47:37 2020

Uses train/test/val_cases/controls df and asserts all present in image dir
and pytorch dataset feeding model

@author: mlim-user - james condon
"""
import os
import pandas as pd
import numpy as np
import argparse
from src.constants import TTSDIR, DATADIR

splits = ['train', 'val', 'test']
cohorts = ['cases', 'controls']

folders = ['train_ims_sf2', 'val_ims_sf2']

def load_split_df(split, cohort):
    """ loads dataframe eg train_cases.csv """
    name = '_'.join([split, cohort, 'df.csv'])
    df = pd.read_csv(os.path.join(TTSDIR, name))
    print('\n\t{0} unique ANs in {1}\n'.format(len(np.unique(df.AN)), name))
    return df

def load_ANs(folder, cohort):
    """ returns unique accession numbers from local dir """
    assert cohort in ['cases', 'controls']
    files = os.listdir(folder)
    if cohort == 'cases':
        string = 'e1' # dicom filename code for cancer episode
        files = [x for x in files if string in x]
    elif cohort == 'controls':
        string = 'e0' # dicom filename code for cancer free episode
        files = [x for x in files if string in x]
    file_ANs = [x.split('-', 3)[-2] for x in files]
    n_eps = len(np.unique(file_ANs))
    print('\n\t{0} unique {1} ANs in {2}'.format(n_eps, cohort, folder))
    return file_ANs

def not_on_file(dfANs, fileANs):
    """ compares dataframe ANs from traintest split with files on disk """
    not_found = []
    for dfAN in dfANs:
        if dfAN not in fileANs:
            not_found.append(dfAN)
    return not_found

def check(split, cohort, folder):
    print(split, cohort)
    df = load_split_df(split, cohort)
    dfANs = df.AN.values
    fileANs = load_ANs(folder, cohort)
    not_found = not_on_file(dfANs, fileANs)
    if len(not_found) == 0:
        print('all data frame episodes in {}'.crop_folder_name)
    
def txt_list_ANs(file):
    t = pd.read_csv(file, sep=" ", header=None)
    textANs = [x.rsplit('-',3)[-3] for x in t.loc[:,0]]
    n = len(np.unique(textANs))
    print('{0} unique ANs in {1}'.format(n, file))
    return np.unique(textANs)
#%%
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input-folder', required=True)
    
    args = parser.parse_args()
