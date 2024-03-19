#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 09:59:23 2019
@author: mlim-user - james

'CLI script to train-test-split and download cases and age-matched-controls .txt list 
(see AIML_data_log for examples)
For subsequent "--from-file" rsync use'

# default 80:10:10 (train, val, test)

"""
import os
import argparse
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
from src.utilities.pickling import pickle_to_file
from src.utilities.all_utils import load_df, age_matcher
from src.constants import BASECOLS, NONG_COLS, cancers, benign_tumours, ca_codes, benign_codes
from src.constants import CSVDIR

class AMCs():
    def __init__(self, CSVDIR, hd_folder, repo_folder):
        _, codes = load_df()
        # master df
#        print('NONG_COLS only')
        print('BASE_COLS only')
#        print('using all coded_megadf cols')
        m_df = pd.read_csv(os.path.join(CSVDIR, 'coded_megadf.csv'),
                          usecols=BASECOLS)
        # slice missing screening accession numbers: n = 3
        m_df = m_df[m_df['SX_ACCESSION_NUMBER.int32()'].astype(int).astype(str).str.len() >= 6]

        m_df.sort_index(axis=1, inplace=True)
        # create col with syntax 'A123456' to match with dm_df:
        if m_df['SX_ACCESSION_NUMBER.int32()'].dtype == float:
            m_df['AN'] = 'A' + m_df['SX_ACCESSION_NUMBER.int32()'].astype(int).astype(str)
        
        assert all(x.startswith('A') for x in m_df['AN'])
        assert all(len(x) >= 7 for x in m_df['AN'])
        assert not any(x.startswith('AA') for x in m_df['AN'])

        # dicom metadata df:
        dm_df = pd.read_csv(
            os.path.join(CSVDIR, 'dcm_master_info_final.csv'),
                         low_memory=False
                        )
        dm_df = dm_df[dm_df['filename'].str.contains('-S-')]
        
        dm_df = dm_df[dm_df['AN'].str.len() > 5]
    
        dm_df = dm_df[dm_df.views.isin(['CC', 'MLO'])] #.reset_index(drop=True)
        dodgy_IDs = [x for x in dm_df['ID'] if '?' in x]
        dm_df = dm_df[~dm_df['ID'].isin(dodgy_IDs)]
            
        dm_df['ID'] = dm_df.ID.str.replace('I','').astype(int)
        self.dm_df = dm_df
        
        assert all(x.startswith('A') for x in dm_df['AN'])
        assert all(len(x) >= 6 for x in dm_df['AN'])
        assert not any(x.startswith('AA') for x in dm_df['AN'])
        
        fnames = [x.split('files/', 1)[-1] for x in self.dm_df.ersa_fp]
        rsync_fps = [x.split('images/', 1)[-1] for x in self.dm_df.ersa_fp]
        par_dirs = [x.split('/', 1)[0] for x in rsync_fps]
        self.dm_df['rsync_fp'] = [x+'/./'+y for x, y in zip(par_dirs, fnames)]

        # save cases df (cancer and benign episodes)
        ca_code_list = list(ca_codes.keys())
        benign_code_list = list(benign_codes.keys())
        ca_df = m_df[m_df['HIST_SNOMED.string()'].isin(ca_code_list)]
        ben_df = m_df[m_df['HIST_SNOMED.string()'].isin(benign_code_list)]
        cas_df = pd.concat([ca_df, ben_df]) # includes women recalled and biopsied with benign tumours
        cas_df['HIST_SNOW_DECODE'] = cas_df['HIST_SNOMED.string()'].apply(lambda x: codes['HIST_SNOMED.string()'].loc[x])
        self.cas_df = cas_df.sort_values('AAS').reset_index(drop=True)
        
        # save IDs of cancer pts:
        Case_IDs = np.unique(cas_df['ID.int32()'].values)
        
        # match master patient_data df (m_df) ANs with dicom metadata df (dm_df)
        m_df = m_df[m_df['AN'].isin(dm_df['AN'])]
        
        # remove women who ever had an in-dataset episode of cancer or benign biopsy / in Ca_IDs:
        m_df = m_df[~m_df['ID.int32()'].isin(Case_IDs)]
        
        """
        CONTROL EXCLUSIONS
         # ever had histology
         # implants
         # last episode without f/up
         # symptoms reported
        """
        # histo / other Biopsy:
        other_bx_eps = m_df[m_df['HIST_OUTCOME.string()'] != 0] # NaN
        print('\n n patients with some other form of biopsy:')
        print(len(pd.unique(other_bx_eps['ID.int32()'])))
        
        m_df = m_df[m_df['HIST_OUTCOME.string()'] == 0]

        # implants
        imp_eps = m_df[m_df['PQ_IMPLANTS.string()'] != 0] # no implants
        print('\n n patients with implants:')
        print(len(pd.unique(imp_eps['ID.int32()'])))
        
        m_df = m_df[m_df['PQ_IMPLANTS.string()'] == 0]
        
        # symptoms:
        symp_eps = m_df[m_df['PQ_SYMPTOMS.string()'] != 3] # no symptoms
        print('\n n patients with symptoms:')
        print(len(pd.unique(symp_eps['ID.int32()'])))
        
        m_df = m_df[m_df['PQ_SYMPTOMS.string()'] == 3]
        
        # Only one episode:
        one_eps = m_df[m_df['Total_Eps'] == 1]
        print('\n n patients with only one episode:')
        print(len(pd.unique(one_eps['ID.int32()'])))
        
        m_df = m_df[m_df['Total_Eps'] > 1]
        
        # Last episode with no follow-up
        last_eps = m_df[m_df['Eps_N_to_last'] == -1]
        print('\n n exams with no_follow_up:')
        print(last_eps.shape[0])
        
        m_df = m_df[m_df['Eps_N_to_last'] < -1]
        print('\n n control pool patients:')
        print(len(pd.unique(m_df['ID.int32()'])))
        
        self.pos_controls = m_df
        
        # Store list of patients who were ever recalled (ER_IDs):
        ER_IDs = m_df[m_df['ASSESSMENT.string()'] == 1]
        ER_IDs = np.unique(ER_IDs['ID.int32()'].values)
        """
        # remove women who were ever recalled:
        self.pos_controls = pos_controls[~pos_controls['ID.int32()'].isin(ER_IDs)]
        """
#        plt.figure()
#        plt.hist(inst.cas_df['AAS'], bins=200, label='cases')
#        plt.hist(inst.pos_controls['AAS'], bins=200, label='all_possible_controls', alpha=0.6)
#        plt.title('Age (controls have at least 1 subsequent normal follow-up)')
#        plt.legend()
#        plt.show()
#        del self.m_df
        gc.collect()
        
        self.hd_folder = hd_folder
        self.repo_folder = repo_folder
               
    def train_val_test_split(self, chrono=False, s_rand=True):
        
        if chrono:
            # split patients chronologically, i.e., train on earliest screeners, test on latest screeners.
            self.pos_controls = self.pos_controls.sort_values('SX_DATE.string()')
            self.cas_df = self.cas_df.sort_values('SX_DATE.string()')
        elif s_rand:
            self.pos_controls = self.pos_controls.sample(frac=1)
            self.cas_df = self.cas_df.sample(frac=1)
        else:
            raise ValueError("either chrono or s_rand must be True")
            
        # split all possible controls / tts:
        self.train_con_df, self.rest_con_df = train_test_split(
                self.pos_controls, 
                train_size=0.8, 
                test_size=0.2, 
                shuffle=False) 
        
        self.val_con_df, self.test_con_df = train_test_split(
                self.rest_con_df, 
                train_size=0.5, 
                test_size=0.5, 
                shuffle=False)
        # cases tts:
        self.train_cas_df, self.rest_cas_df = train_test_split(
                self.cas_df, 
                train_size=0.7, 
                test_size=0.3, 
#                shuffle=False,
                stratify=self.cas_df['AX_WU_DOM_CAT_LESA_EV1.string()'] # splits equal numbers of Calc, Stellate, discrete mass, arch. distortion
                ) 
        
        self.val_cas_df, self.test_cas_df = train_test_split(
                self.rest_cas_df, 
                train_size=0.5, 
                test_size=0.5, 
#                shuffle=False,
                stratify=self.rest_cas_df['AX_WU_DOM_CAT_LESA_EV1.string()'] # splits equal numbers of Calc, Stellate, discrete mass, arch. distortion
                )
        
    def match(self, n_controls, folder):
        # age matcher returns randomly shuffled df of cases and controls but
        # also saves just controls df to self.CSVDIR (in ascending Age At Screening (AAS))
        train_cascondf = age_matcher(self.train_cas_df, self.train_con_df, folder, split='train', n_controls=n_controls, control='No_Ca')
        train_cascondf = train_cascondf.drop('index',axis=1)
        self.train_cascondf = train_cascondf.reset_index(drop=True)
        
        val_cascondf = age_matcher(self.val_cas_df, self.val_con_df, folder, split='val', n_controls=n_controls, control='No_Ca')
        val_cascondf = val_cascondf.drop('index',axis=1)
        self.val_cascondf = val_cascondf.reset_index(drop=True)
        
        test_cascondf = age_matcher(self.test_cas_df, self.test_con_df, folder, split='test', n_controls=n_controls, control='No_Ca')
        test_cascondf = test_cascondf.drop('index',axis=1)
        self.test_cascondf = test_cascondf.reset_index(drop=True)
        
        self.train_con_df = pd.read_csv(os.path.join(folder, 'train_age_matched_controls.csv'))
        print("train controls' break down of Eps_to_last:")
        print('\n', self.train_con_df['Eps_N_to_last'].value_counts(normalize=True))
        
        self.val_con_df = pd.read_csv(os.path.join(folder, 'val_age_matched_controls.csv'))
        print("train controls' break down of Eps_to_last:")
        print('\n', self.train_con_df['Eps_N_to_last'].value_counts(normalize=True))
        
        self.test_con_df = pd.read_csv(os.path.join(folder, 'test_age_matched_controls.csv'))
        print("train controls' break down of Eps_to_last:")
        print('\n', self.train_con_df['Eps_N_to_last'].value_counts(normalize=True))
        
        for df in [self.train_con_df, self.val_con_df, self.test_con_df]:
            assert all([x.startswith('A') for x in df.AN])
        
    def save(self, folder):
        dfs = [
                self.train_con_df,
                self.val_con_df,
                self.test_con_df,
                self.train_cas_df,
                self.val_cas_df,
                self.test_cas_df
                ]
        names = ['train_controls',
                 'val_controls',
                 'test_controls',
                 'train_cases',
                 'val_cases',
                 'test_cases'
                 ]
        for name, df in zip(names, dfs):
            print('\n', name, df.head())
            df.to_csv(os.path.join(folder, name + '_df.csv'), index=False)
        
    def slice_dcm_meta(self):
        # train_controls:
        self.train_con_ANs = self.train_con_df['AN'].values   
        self.train_con_dm_df = self.dm_df[self.dm_df.AN.isin(self.train_con_ANs)]
        # val_controls:
        self.val_con_ANs = self.val_con_df['AN'].values   
        self.val_con_dm_df = self.dm_df[self.dm_df.AN.isin(self.val_con_ANs)]
        
        # test_controls:
        self.test_con_ANs = self.test_con_df['AN'].values   
        self.test_con_dm_df = self.dm_df[self.dm_df.AN.isin(self.test_con_ANs)]
        
        # train_cases:
        self.train_cas_ANs = self.train_cas_df['AN'].values   
        self.train_cas_dm_df = self.dm_df[self.dm_df.AN.isin(self.train_cas_ANs)]
        # val_cases:
        self.val_cas_ANs = self.val_cas_df['AN'].values   
        self.val_cas_dm_df = self.dm_df[self.dm_df.AN.isin(self.val_cas_ANs)]
        # test_cases:
        self.test_cas_ANs = self.test_cas_df['AN'].values   
        self.test_cas_dm_df = self.dm_df[self.dm_df.AN.isin(self.test_cas_ANs)]
        
    def vis_split_finding_dists(self):
        """ visualise proportions of significant work-up findings
        (calcificiation, massess, arch. distortion) """
        self.train_cas_df['AX_WU_DOM_CAT_LESA_EV1.string()'].value_counts().plot(autopct='%1.1f%%', kind='pie')
        self.val_cas_df['AX_WU_DOM_CAT_LESA_EV1.string()'].value_counts().plot(autopct='%1.1f%%', kind='pie')
        self.test_cas_df['AX_WU_DOM_CAT_LESA_EV1.string()'].value_counts().plot(autopct='%1.1f%%', kind='pie')
    
    def save_ANs_and_dm_dfs(self, folder):
        # save list of ANs and dicom metadata dfs for subsequent file renaming and preprocessing.
        AN_lists = [
                self.train_con_ANs,
                self.val_con_ANs,
                self.test_con_ANs, 
                self.train_cas_ANs, 
                self.val_cas_ANs,
                self.test_cas_ANs
                ]
        dfs = [
                self.train_con_dm_df,
                self.val_con_dm_df,
                self.test_con_dm_df,
                self.train_cas_dm_df,
                self.val_cas_dm_df,
                self.test_cas_dm_df
                ]
        names = ['train_controls',
                 'val_controls',
                 'test_controls',
                 'train_cases',
                 'val_cases',
                 'test_cases'
                 ]
        for AN_list, name, df in zip(AN_lists, names, dfs):
            print('\n', AN_list[:4], name, df.head())
            pickle_to_file(
                    file_name=os.path.join(folder, name + '_ANs.pkl'),
                    data=AN_list)
            df.to_csv(os.path.join(folder, name + '_dm_df.csv'), index=False)
            
    def save_txt_lists(self):
        
        if not os.path.exists(self.repo_folder):
            os.mkdir(self.repo_folder)
            
        train_con_path = os.path.join(self.repo_folder, 'train_controls.txt')
        val_con_path = os.path.join(self.repo_folder, 'val_controls.txt')
        test_con_path = os.path.join(self.repo_folder, 'test_controls.txt')
        train_case_path = os.path.join(self.repo_folder, 'train_cases.txt')
        val_case_path = os.path.join(self.repo_folder, 'val_cases.txt')
        test_case_path = os.path.join(self.repo_folder, 'test_cases.txt')
        # Age-Matched Controls
        np.savetxt(
                    fname=train_con_path,
                    X=list(self.train_con_dm_df.rsync_fp),
                    fmt='%s'
                    )
        np.savetxt(
                    fname=val_con_path,
                    X=list(self.val_con_dm_df.rsync_fp),
                    fmt='%s'
                    )
        np.savetxt(
                    fname=test_con_path,
                    X=list(self.test_con_dm_df.rsync_fp),
                    fmt='%s'
                    )
        # Cases
        np.savetxt(
                    fname=train_case_path,
                    X=list(self.train_cas_dm_df.rsync_fp),
                    fmt='%s'
                    )
        np.savetxt(
                    fname=val_case_path,
                    X=list(self.val_cas_dm_df.rsync_fp),
                    fmt='%s'
                    )
        np.savetxt(
                    fname=test_case_path,
                    X=list(self.test_cas_dm_df.rsync_fp),
                    fmt='%s'
                    )
        #%%
def main(args):
    
    inst = AMCs(
           CSVDIR,
           args.hd_folder,
           args.repo_folder
           )
    if args.split_chronologically:
        inst.train_val_test_split(chrono=True, s_rand=False)
    elif args.split_randomly:
        inst.train_val_test_split(chrono=False, s_rand=True)
    inst.match(n_controls=args.AMCs_per_case, folder=args.hd_folder)
    inst.save(folder=args.hd_folder)
    inst.slice_dcm_meta()
    inst.vis_split_finding_dists()
    inst.save_ANs_and_dm_dfs(args.hd_folder)
    inst.save_txt_lists()
    print('Preparing folders for downloads')
    paths = ['train/controls',
             'train/cases',
             'val/controls',
             'val/cases',
             'test/controls',
             'test/cases'
             ]
    print("Do you want to make dirs for train/val/test for cases/controls in:")
    print(args.hd_folder)
    print("('y'/'n')")
    answ = input()
    if answ == 'y':
        for folder in paths:
            if not os.path.exists(os.path.join(args.hd_folder, folder)):
                os.makedirs(os.path.join(args.hd_folder, folder))
            else:
                print(os.path.join(args.hd_folder, folder), 'already exists')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLI script to download age-matched-controls .txt list for subsequent "--from-file" rsync use')
#    parser.add_argument('--CSVDIR', type=str, default='/home/mlim-user/Documents/james/tempdir/')
    parser.add_argument('--repo-folder', type=str, default='/data/james/BSSA_images', help='directory within repo for record of .txt dicom tts lists')
    parser.add_argument('--hd-folder', type=str, required=True, help='Local directory to save dicom metadata frames and Accession Number .pkl and setup dirs for downloads')
    parser.add_argument('--AMCs-per-case', type=int, required=True)
    parser.add_argument('--split-chronologically', action='store_true')
    parser.add_argument('--split-randomly', action='store_true')

    args = parser.parse_args()
    
    assert os.path.exists(args.repo_folder)
    assert os.path.exists(args.hd_folder)

    print('\n******* \n Are you SURE you want to potentially overwrite')
    print('existing AN lists and dfs in:\n', args.hd_folder, '?')
    print('\nand .txt lists in:\n', args.repo_folder, '?\n(enter)')
    print('\n(age matcher has some randomness to controls, so will be different from any existing lists)')
    print('\n************')
    input()    

    main(args)
    
    print('\n\tFinished.')

