#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 09:08:44 2020

Loads one dicom per Accession number in dicom directory.
Saves manufacturer to train-test-split (TTS) logs dataframe.

@author: James Condon
"""
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pydicom
from src.utilities.all_utils import data_to_df

SPLITDIR = '/nvme/james/train_ims_master/'
OUTDIR = '/data/james/NYU_retrain/TTS_logs/dfs_with_vendors/5050/train_df.csv'

df, files = data_to_df(
        data=os.path.join(SPLITDIR, 'data_sf2.pkl'),
        return_files=True)
#%%
ca = df[df['cancer_ep'] == 1]
controls = df[(df['cancer_ep'] == 0) & (df['benign_tumour_ep'] == 0)]
plt.figure()
plt.title('Age in training dataset: 44.1% incidence of malignancy')
plt.hist([ca['AAS'], controls['AAS']], density=False, bins=50)
plt.legend(['cases', 'controls'])

plt.ylabel('n')

plt.xlabel('Age')
plt.show()
#%%
    
for i, file in enumerate(tqdm(files)): # eg 'I666171-E2-A194357-S-i3.dcm'
    AN = file.split('-', 3)[-2] # eg 'A194357'
    # store index
#    print(AN)
    try:
        ix = df.index[df['AN'] == AN].values[0]
    except IndexError as f:
        print('\n', f)
        print('Accession number {} not in df. Skipped...'.format(AN))
        pass
        
    dcm = pydicom.read_file(os.path.join(SPLITDIR, file)+'.dcm')
    df.at[ix, 'vendor'] = dcm.Manufacturer
    # take out all files with that AN from remaining pool (as all images/episode will have same vendor)
    files = [x for x in files if AN not in x]
    
df.to_csv(OUTDIR, index=False)
