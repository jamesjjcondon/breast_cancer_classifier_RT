#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 11:39:36 2020

Slices out cases and controls already in balanced train-val splits
to acheive validation sets with real-world imbalance

"We have 5,832 exams with at least one biopsy
performed within 120 days of the screening mammogram.
Among these, biopsies confirmed malignant findings for 985
(8.4%) breasts and benign findings for 5,556 (47.6%) breasts.
234 (2.0%) breasts had both malignant and benign findings"
    - Wu et al, 2019

- test_set has 425 carcinoma episodes (excl. LCIS). 
- therefore to simulate NYU cancer episode incidence (8.4% of breasts / 16.8% of pts), need 2530 normal episodes
 or for real-world incidence:
     # (0.7%) 60714 normal episodes
     # (1%) 42500 normal episodes
(not previously trained or tested on)

@author: James Condon
"""
import os
import numpy as np
import pandas as pd
import random
from src.utilities.all_utils import age_matcher_lite, load_df, check_years, compare_ttsdf_and_examlist
from src.utilities import pickling
from src.constants import BASECOLS, CSVDIR, REPODIR, TTSDIR, TESTDIR, ca_codes, benign_codes
#%%
def main(n_controls, out_dir):
    """ Uses ratio of cases to controls. k = n_cases * n_controls. k extra controls 
    Saves df and text list in TTSDIR and out_dir respectively """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Take out control patients already used in train, val and test sets:
    train_con = pd.read_csv(os.path.join(TTSDIR, 'train_controls_df.csv')) #['ID'].values
    train_IDs = np.unique(train_con['ID.int32()'].values)
    val_con = pd.read_csv(os.path.join(TTSDIR, 'val_controls_df.csv')) #['ID'].values
    val_IDs = np.unique(val_con['ID.int32()'].values)
    
    #test_cas = pd.read_csv(os.path.join(TTSDIR, 'test_cases_df.csv')) #)['ID'].values
    test_con = pd.read_csv(os.path.join(TTSDIR, 'test_controls_df.csv')) #)['ID'].values
    test_IDs = np.unique(test_con['ID.int32()'].values)
    
    #%% list of patient IDs already in 5050 train-test split: (n=14 455)    
    #results = check_years()   
    # Store list of pt IDs already in control splits:
    IDs = np.concatenate([train_IDs, val_IDs, test_IDs]) # can re-include , test_control patients])
    
    #%%
    # get more controls to simulate real world incidence:
    df, _ = load_df(coded=True)
    
    # Remove women already in train/val/test set:
    df = df[~df['ID.int32()'].isin(IDs)]
    
    """
        CONTROL EXCLUSIONS
         # ever had histology
         # last episode without f/up
         # implants
         # symptoms reported
    """
    df = df[
            (df['HIST_OUTCOME.string()'] == 0) & \
            (df['HIST_SNOMED.string()'] == 44) & \
            (df['Total_Eps'] > 1) & \
            (df['Eps_N_to_last'] < -1) & \
            (df['PQ_IMPLANTS.string()'] == 0) & \
            (df['PQ_SYMPTOMS.string()'] == 3) 
            ]

    # remove women who ever had an episode of cancer / not in Ca_IDs:
    ca_IDs = np.unique(df[df['ca_pt'] == 1]['ID.int32()'])
    df = df[~df['ID.int32()'].isin(ca_IDs)]
    
    control_pts = np.unique(df['ID.int32()'])
    n = len(control_pts)
    print('\n\tall possible controls (patients) now', n)
    
    """
    # test_set has 425 carcinoma episodes (excl. LCIS). 
    # therefore to simulate NYU cancer episode incidence, 
    # need 2530 normal episodes (8.4% of breasts / 16.8% of pts)
    # or 42500 normal episodes for real-world incidence (approx 1%).
    # (not previously trained or tested on)
    below includes benign cases
    """
    #%% extra control IDs (random)
#    xIDs = random.sample(set(control_pts), k=k)
#    df = df[df['ID.int32()'].isin(xIDs)] # pt-wise so multiple eps / pt
#    df.drop_duplicates(subset='ID.int32()', keep='last', inplace=True) # only one ep / pt (the most recent one - reverse order)
#    
    # age_matched:
    test_cas_df = pd.read_csv(
            os.path.join(TTSDIR, 'test_cases_df.csv'))
    exam_list = pickling.unpickle_from_file(
            os.path.join(TESTDIR, 'data_sf2.pkl'))
    test_cas_df = compare_ttsdf_and_examlist(test_cas_df, exam_list)
    
    test_case_control_df, controls_df = age_matcher_lite(cases=test_cas_df,
                                                         pos_controls=df,
                                                         folder=out_dir,
                                                         n_controls=n_controls)
    
    x_con_ANs = np.unique(controls_df['AN'].values)
    dmdf = pd.read_csv(os.path.join(CSVDIR, 'dcm_master_info_final.csv'))
    dmdf = dmdf[dmdf['views'].isin(['CC', 'MLO'])]
    xcondf = dmdf[dmdf['AN'].isin(x_con_ANs)]
    k = test_cas_df.shape[0] * n_controls
    
    xcondf.to_csv(os.path.join(TTSDIR, '{}_controls_dmdf.csv'.format(k)), index=False)
    controls_df.to_csv(os.path.join(TTSDIR, '{}_controls_df.csv'.format(k)), index=False)
    test_case_control_df.to_csv(
            os.path.join(TTSDIR, 'test_cases_{}_controls_df.csv'.format(k)), index=False)
    np.savetxt(
            fname=os.path.join(out_dir, '{}_extra_controls.txt'.format(k)),
            X=list(xcondf.rsync_fps),
            fmt='%s'
            )
    return print('\nDone.')

if __name__ == "__main__":
    out_dir = os.path.join(REPODIR, 'AIML_data_log/extra_controls_for_real_incidence')
    main(
            n_controls=6,
            out_dir=out_dir
            )
    