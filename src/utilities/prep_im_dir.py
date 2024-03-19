#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:37:32 2020

Changes dicom names to include cancer label code
Constructs Wu/NYU-style patient-wise dictionary
(see 'exam_list_before_cropping.pkl' 
@ https://github.com/nyukat/breast_cancer_classifier/tree/master/sample_data )
@author: James Condon
"""
import os
import pandas as pd
from tqdm import tqdm
from src.constants import CSVDIR, DATADIR, NVMEDIR
from src.utilities.all_utils import nyu_exam_list
   #%%
def change_dcm_names(IMDIR):
    print('\n\tLoading dfs...')
    df = pd.read_csv(os.path.join(CSVDIR, 'super.csv'))
    print(df.head())
    print('\ndf shape:\n',df.shape)
    # slice out files other than .dcm:
    dicoms = [x for x in sorted(os.listdir(IMDIR)) if x.endswith('.dcm')]
    print('\n{} dicoms found in'.format(str(len(dicoms))) + IMDIR)
    df = df[df['filename'].isin(dicoms)]
    print('\ndf shape:\n', df.shape)
    print('dropping duplicate filenames...')
    df = df[~df.duplicated('filename')].sort_values('filename')
    
    assert dicoms == df['filename'].values.tolist(), "Mismatch between dicoms in {} and old filenames.".format(IMDIR)
    dicoms = [os.path.join(IMDIR, x) for x in dicoms]

    assert df.shape[0] == len(dicoms), "Mismatch between dicoms in {} and old filenames.".format(IMDIR)
    
#    old_fps = sorted([os.path.join(IMDIR, x) for x in df['filename']])
    new_fps = sorted([os.path.join(IMDIR, x) for x in df['new_names']])
    
    for old_fp, new_fp in tqdm(zip(dicoms, new_fps), 'changing names'):
#        print('\n', old_fp)
#        print(new_fp)
        if os.path.exists(new_fp):
            continue
        old_name = os.path.split(old_fp)[-1]
        new_name = os.path.split(new_fp)[-1]
#        print(old_name)
#        print(new_name)
        assert new_name.startswith(old_name.split('.dcm')[0])
        """ rename dicoms """
        os.rename(old_fp, new_fp)
        
def main(IMDIR):
    #change_dcm_names(IMDIR)
    nyu_exam_list(
            im_folder=IMDIR,
            dic_fp=os.path.join(os.path.split(IMDIR)[0],'pre_crop'),
            file_format='.dcm'
            )   
    
if __name__ == "__main__":
    IMDIR = os.path.join(DATADIR, 'test_ims_master/renamed_dicoms')
    OUTDIR = os.path.join(DATADIR, 'test_ims_master/test_ims_mini')
    assert os.path.exists(IMDIR), os.path.exists(OUTDIR)
    print("\n\tChanging dicom names in {} to include code for patient, episode and image cancer labels..".format(IMDIR))
    print("*********\nContinue with \n{} \nand\n".format(IMDIR) + CSVDIR, '?')
    print('(Enter)')
    input()
    main(IMDIR)