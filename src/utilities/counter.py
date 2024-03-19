#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:36:58 2020

@author: mlim-user
"""
import os
import numpy as np
import pandas as pd
from src.constants import CSVDIR, ca_codes, benign_codes
from src.utilities.all_utils import load_df

# df = pd.read_csv(os.path.join(CSVDIR, 'super.csv'))

#%%
dmdf = pd.read_csv(os.path.join(CSVDIR, 'dcm_master_info_final.csv'))
dmdf.drop_duplicates(subset='filename', inplace=True)
dmdf.shape

screen_ims = [x for x in dmdf.filename if 'S'  in x]

print('\n', len(screen_ims), ' Screening images in dicom metadata dataframe (dmdf).')
print('removed assessmnets')

dmdf = dmdf[dmdf.filename.isin(screen_ims)]
#%%
dirty_pt_rows = [x for x in dmdf.ID if '?' in x]
clean_pt_rows = [x for x in dmdf.ID if '?' not in x] # unique patient IDs
print('\n', len(clean_pt_rows), ' clean and {} dirty pt rowss in dmdf'.format(len(dirty_pt_rows)))
#%
dmdf = dmdf[dmdf['ID'].isin(clean_pt_rows)]
    #%%

nonMLOCC = [x for x in dmdf.views if x not in ['MLO', 'CC']]
MLOCC = [x for x in dmdf.views if x in ['MLO', 'CC']]

removed_ims = len(nonMLOCC) + len(dirty_pt_rows)
print('\n{} removed ims'.format(removed_ims))
dmdf = dmdf[dmdf['views'].isin(MLOCC)]
#%%
n_pts = np.unique(dmdf.ID)
n_screen_eps = np.unique(dmdf.AN)
#%%
cdf, codes = load_df()
cdf['AN'] = cdf['SX_ACCESSION_NUMBER.int32()'].apply(lambda x: 'A' + str(int(x)))

# save cases df (cancer and benign episodes)
ca_code_list = list(ca_codes.keys())
benign_code_list = list(benign_codes.keys())
ca_df = cdf[cdf['HIST_SNOMED.string()'].isin(ca_code_list)]
ben_df = cdf[cdf['HIST_SNOMED.string()'].isin(benign_code_list)]
cas_df = pd.concat([ca_df, ben_df]) # includes women recalled and biopsied with benign tumours
cas_df['HIST_SNOW_DECODE'] = cas_df['HIST_SNOMED.string()'].apply(lambda x: codes['HIST_SNOMED.string()'].loc[x])
cas_df = cas_df.sort_values('AAS').reset_index(drop=True)

# save IDs of cancer pts:
Case_IDs = np.unique(cas_df['ID.int32()'].values)# working dataframe:

df = cdf[cdf['AN'].isin(dmdf.AN)]

#%%
"""
CONTROL EXCLUSIONS
         # ever had histology
         # last episode without f/up
         # implants
         # symptoms reported
        """
df = df[
        (df['HIST_OUTCOME.string()'] == 0) & \
        (df['Total_Eps'] > 1) & \
        (df['Eps_N_to_last'] < -1) & \
        (df['PQ_IMPLANTS.string()'] == 0) & \
        (df['PQ_SYMPTOMS.string()'] == 3) 
        ]
# remove women who ever had an episode of cancer / not in Ca_IDs:
df = df[~df['ID.int32()'].isin(Case_IDs)]

n_control_pool_pts = len(np.unique(df['ID.int32()']))
print('\n', n_control_pool_pts, 'patients in control pool.')
    #%%
len(np.unique(df['ID.int32()']))
len(np.unique(df.AN))

dmdf[dmdf.AN.isin(df.AN)].shape

cas_df.shape

np.unique(cas_df['ID.int32()'])
len(np.unique(cas_df['ID.int32()']))
len(np.unique(cas_df['AN']))
casdmdf = dmdf[dmdf.AN.isin(cas_df.AN)]
#clean pts:
# print('\nMLO CC images:', len(np.unique(wdf.filename)))

# #%%
# print('\n number of patients with mlo and CC and clean IDs:')
# print(len(np.unique(wdf.ID)))
# #%%
# print('\nEpisodes:', len(np.unique(wdf.AN)))