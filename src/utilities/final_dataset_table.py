#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 20:12:52 2020

@author: mlim-user
"""
import os
import numpy as np
import pandas as pd
from src.utilities.all_utils import data_to_df, load_df, decode_col
from src.constants import DATADIR

# load data and merge with vendor data (previously saved seprately):
train = data_to_df(
        os.path.join(DATADIR, 'train_ims_master/data_sf2.pkl'))
train_vends = pd.read_csv(
        os.path.join(DATADIR, 'TTS_logs/dfs_with_vendors/balanced/train_df.csv'),
        usecols=['vendor', 'AN'])

dftr = pd.merge(train, train_vends)

val = data_to_df(
        os.path.join(DATADIR, 'val_ims_master/data_sf2.pkl'))
val_vends = pd.read_csv(
        os.path.join(DATADIR, 'TTS_logs/dfs_with_vendors/balanced/val_df.csv'),
        usecols=['vendor', 'AN'])
dfv = pd.merge(val, val_vends)


test_b = data_to_df(
        os.path.join(DATADIR, 'test_ims_master/data_sf2.pkl'))
test_vends = pd.read_csv(
        os.path.join(DATADIR, 'TTS_logs/dfs_with_vendors/balanced/test_df.csv'),
        usecols=['vendor', 'AN'])
dftestb = pd.merge(test_b, test_vends)


test_n = data_to_df(
        os.path.join(DATADIR, 'test_ims_master/NY_16-8pc_incidence/data_sf2.pkl'))
testn_vends = pd.read_csv(
        os.path.join(DATADIR, 'TTS_logs/dfs_with_vendors/NY_16-8pc_incidence/test_cases_2814_controls_df_with_vendor.csv'),
        usecols=['vendor', 'AN'])

dftestn = pd.merge(test_n, testn_vends)
#%%
train_ca = dftr[dftr['ca_pt'] == 1]
train_ben = dftr[dftr['benign_pt'] == 1]
train_cons = dftr[(dftr['ca_pt'] == 0) & (dftr['benign_pt'] == 0)]

val_ca = dfv[dfv['ca_pt'] == 1]
val_ben = dfv[dfv['benign_pt'] == 1]
val_cons = dfv[(dfv['ca_pt'] == 0) & (dfv['benign_pt'] == 0)]

test_ca = dftestb[dftestb['ca_pt'] == 1]
test_ben = dftestb[dftestb['benign_pt'] == 1]
#balanced:
test_consb = dftestb[(dftestb['ca_pt'] == 0) & (dftestb['benign_pt'] == 0)]
#nyu:
test_consn = dftestn[(dftestn['ca_pt'] == 0) & (dftestn['benign_pt'] == 0)]

all_ca = pd.concat([train_ca, val_ca, test_ca], axis=0)
all_ben = pd.concat([train_ben, val_ben, test_ben], axis=0)
train_cases = pd.concat([train_ca, train_ben], axis=0)
val_cases = pd.concat([val_ca, val_ben], axis=0)
test_cases = pd.concat([test_ca, test_ben], axis=0)

all_cases = pd.concat([train_cases, val_cases, test_cases], axis=0)

all_cons_b = pd.concat([train_cons, val_cons, test_consb], axis=0)
all_cons_n = pd.concat([train_cons, val_cons, test_consn], axis=0)

#%%
_, codes = load_df()

#%%
def print_value_counts(col):
    stage_codes = decode_col(codes, col)
    print('\ntotal dataset:')
    print((all_ca[col].value_counts(dropna=False)).sort_index())
    print(all_ca[col].value_counts(dropna=False, normalize=True).sort_index())
    round_pcs = np.round(all_ca[col].value_counts(dropna=False, normalize=True)*100, 3).sort_index()
    print(round_pcs)
    print(stage_codes)
    
    print('\n train:')
    print((train_ca[col].value_counts(dropna=False)).sort_index())
    print(train_ca[col].value_counts(dropna=False, normalize=True).sort_index())
    round_pcs = np.round(train_ca[col].value_counts(dropna=False, normalize=True)*100, 3).sort_index()
    print(round_pcs)
    print(stage_codes)
    
    print('\n vals:')
    print((val_ca[col].value_counts(dropna=False)).sort_index())
    print(val_ca[col].value_counts(dropna=False, normalize=True).sort_index())
    round_pcs = np.round(val_ca[col].value_counts(dropna=False, normalize=True)*100, 3).sort_index()
    print(round_pcs)
    print(stage_codes)
    
    print('\n test:')
    print((test_ca[col].value_counts(dropna=False)).sort_index())
    print(test_ca[col].value_counts(dropna=False, normalize=True).sort_index())
    round_pcs = np.round(test_ca[col].value_counts(dropna=False, normalize=True)*100, 3).sort_index()
    print(round_pcs)
    print(stage_codes)

print_value_counts('HIST_STAGE.string()')
#%%
def print_snomed_counts():
    snomed_codes = decode_col(codes, 'HIST_SNOMED.string()')
    print('\ntotal dataset:\n')
    counts = all_cases['HIST_SNOMED.string()'].value_counts(dropna=False).sort_index()
    round_pcs = np.round(all_cases['HIST_SNOMED.string()'].value_counts(dropna=False, normalize=True)*100, 3).sort_index()
    for frame in [counts, round_pcs]:
        frame.index = [snomed_codes.get(x) for x in frame.index]
    print(counts)
    print(round_pcs)

    
    print('\n train:\n')
    counts = train_cases['HIST_SNOMED.string()'].value_counts(dropna=False).sort_index()
    round_pcs = np.round(train_cases['HIST_SNOMED.string()'].value_counts(dropna=False, normalize=True)*100, 3).sort_index()
    for frame in [counts, round_pcs]:
        frame.index = [snomed_codes.get(x) for x in frame.index]
    print(counts)
    print(round_pcs)
    
    print('\n vals:\n')
    counts = val_cases['HIST_SNOMED.string()'].value_counts(dropna=False).sort_index()
    round_pcs = np.round(val_cases['HIST_SNOMED.string()'].value_counts(dropna=False, normalize=True)*100, 3).sort_index()
    for frame in [counts, round_pcs]:
        frame.index = [snomed_codes.get(x) for x in frame.index]
    print(counts)
    print(round_pcs)
    
    print('\n test:\n')
    counts = test_cases['HIST_SNOMED.string()'].value_counts(dropna=False).sort_index()
    round_pcs = np.round(test_cases['HIST_SNOMED.string()'].value_counts(dropna=False, normalize=True)*100, 3).sort_index()
    for frame in [counts, round_pcs]:
        frame.index = [snomed_codes.get(x) for x in frame.index]
    print(counts)
    print(round_pcs)
    
print_snomed_counts()
#%%
print('\nall cancer vendors:')
print('\n', all_ca['vendor'].value_counts(dropna=False))
print(all_ca['vendor'].value_counts(dropna=False, normalize=True))

print('\nall benign vendors:')
print('\n', all_ben['vendor'].value_counts(dropna=False))
print(all_ben['vendor'].value_counts(dropna=False, normalize=True))

print('\nall control balanced vendors:')
print('\n', all_cons_b['vendor'].value_counts(dropna=False))
print(all_cons_b['vendor'].value_counts(dropna=False, normalize=True))

print('\nall control nyu vendors:')
print('\n', all_cons_n['vendor'].value_counts(dropna=False))
print(all_cons_n['vendor'].value_counts(dropna=False, normalize=True))

print('\nall malignancies, all benign lesions age at screening mean and std:')
print(all_ca.AAS.mean())
print(all_ca.AAS.std())
print(all_ben.AAS.mean())
print(all_ben.AAS.std())

print('\n all controls balanced age at screening mean and std:')
print(all_cons_b.AAS.mean())
print(all_cons_b.AAS.std())

print('\n nyu controls age:')
print(all_cons_n.AAS.mean())
print(all_cons_n.AAS.std())

#%%
for df in [train_ca, val_ca, test_ca, 
           train_ben, val_ben, test_ben,
           train_cons, val_cons, test_consb]:
    print('\n')
    print(df['vendor'].value_counts(dropna=False))
    print(df.AAS.std(), df.AAS.mean())
    print(df['PQ_HRT.string()'].value_counts(dropna=False))