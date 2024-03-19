#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:59:40 2019

@author: mlim-user
"""
import sys
import os

from tqdm import tqdm
import time
import gc
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import torch
import pickle
import pydicom

from src.utilities.pickling import unpickle_from_file
import src.utilities.pickling as pickling

from src.constants import CSVDIR as csvdir
from src.constants import BASECOLS as basecols
from src.constants import ca_codes, benign_codes, VIEWS

csvloc = csvdir

def add_ep_before(df, m_df):
    """ For a df from combined_extract.csv with one row per patient, for each row / patient / episode, 
    add the episode prior """
    for ID in tqdm(df['ID.int32()'].values, desc='adding ep prior'):
        Ep_to_last = df[df['ID.int32()'] == ID]['Eps_N_to_last'].values[0]
        row = m_df[(m_df['ID.int32()'] == ID) & (m_df['Eps_N_to_last'] == (Ep_to_last -1))]

        if len(row) > 0:
            df = pd.concat([df, row])

    return df


def add_size_cols(df):
    """ Generate pathology size bins based on original combined_extract"""
    assert df.shape[1] < 10, "use a subset to save RAM"
    inv_size_cols = [x for x in df.columns if 'MALIG_INV_SIZE' in x]
    df['invasive_>15mm'] = df[inv_size_cols].apply(lambda x: int(any(x > 15)), axis=1)
    return df



def age_matcher(cases, pos_controls, folder, split, n_controls=1, control='no_Ca'):
    """
    Takes any df, treated as cases, with 'HIST_OUTCOME.string()' and AAS (age at screening) 
    column and returns df with controls. 
    For cases, most recent episode is used for matching.
    For controls, if matching to "normal" match is two second last episode. 
    Saves:
        1. Just matched controls df
        2. Shuffled case_control_df 
            to 'folder'.
    Args:
        - df1 = cases
        - pos_controls = all potential controls
        - folder = dir
        - control = patient group to use as controls for age-matching eg no cancer, false_positives
    """
    print('\nAge-matching cases and controls...')
    # Make sure cases are cancer only: update for constants.cancer
 
    controls = pos_controls[pos_controls['Eps_N_to_last'] < -1] # match to most recent episode (which isn't last episode)
    
    # Sort cases and controls in ascending Age At Screening (AAS)
    controls = controls.sort_values('AAS').reset_index(drop=False)
    cases = cases.sort_values('AAS').reset_index(drop=False)
    
    del pos_controls
    gc.collect() # save some RAM
    
    matches = [] 
    for i, age in enumerate(tqdm(cases['AAS'], desc='Age-matching')):
        for _ in range(n_controls):
            if len(matches) == cases.shape[0]:
                break
            
            match = controls[controls['AAS'] == age] # slice all controls with same age
            n_matches = match.shape[0] # store n of matches
            if n_matches == 0:
                dif = (controls['AAS'] - age) / -1 # calc difference between case and all controls
                ix = dif.idxmin() # get index of smalleset difference
                match = controls.iloc[ix,:] # slice for the first match (if > 1)
            else:
                ix = np.random.randint(0,n_matches) # choose one randomly
                match = match.iloc[ix,:]

            matches.append(match)
            controls = controls[controls['ID.int32()'] != match['ID.int32()']] # drop patient from control pool so not re-used
            controls = controls.reset_index(drop=True)

    match_cons = pd.concat(matches, axis=1).T.reset_index(drop=True) # store all matches

    # save df of just matched episodes (sorted in ascending Age At Screening (AAS))
    print('{} match_cons.csv shape:'.format(split), match_cons.shape)
    match_cons.to_csv(os.path.join(folder, split+'_age_matched_controls.csv'), index=False)
    
    plt.figure(figsize=(10,6))
    plt.title('Age dsitributions of {} cases and matched controls'.format(split))
    plt.xlabel('age')
    plt.ylabel('counts')
    plt.hist([cases.AAS, match_cons.AAS], 
             density=True,
             bins=50) #visualise distribution
    plt.legend(['cases', 'controls'])
    name = '{} age_cases_controls'.format(split) + time.strftime("_%d_%b_%Y_%H%M") + '.png'
    plt.savefig(os.path.join(folder, name))
    # df of randomly sorted (frac=1), mixed cases and matched controls' episodes
    case_control_df = pd.concat([cases, match_cons], axis=0).sample(frac=1) # stack cases and age-matched controls and shuffle 
    case_control_df.to_csv(
            os.path.join(folder, split + '_age_matched_case_control_df.csv'),
            index=False
            )
    return case_control_df

def age_matcher_lite(cases, pos_controls, folder, n_controls=1, control='no_Ca'):
    """
    Takes any df, treated as cases, with 'HIST_OUTCOME.string()' and AAS (age at screening) 
    column and returns df with controls. 

    Args:
        - cases
        - pos_controls = all potential controls
        - folder = dir to save age dist. vis
        - control = patient group to use as controls for age-matching eg no cancer, false_positives
    """
    print('\nAge-matching cases and controls...')
    # Assert not using last episode (with no f/up) for possible controls:   
    pos_controls = pos_controls[pos_controls['Eps_N_to_last'] < -1] # match to most recent episode (which isn't last episode)
    
    # Sort cases and pos_controls in ascending Age At Screening (AAS) and use only essential cols (saves time)
    pos_controls_4 = pos_controls.sort_values('AAS').reset_index(drop=False)[['ID.int32()', 'AN', 'AAS', 'Eps_N_to_last']]
    cases = cases.sort_values('AAS').reset_index(drop=False) #[['ID.int32()', 'AN', 'AAS', 'Eps_N_to_last']]
    
    matches = [] 
    for i, age in enumerate(tqdm(cases['AAS'], desc='Age-matching')):
        for _ in range(n_controls):
            if len(matches) == cases.shape[0]*n_controls:
                break
            match = pos_controls_4[pos_controls_4['AAS'] == age] # slice all pos_controls_4 with same age
            n_matches = match.shape[0] # store n of matches
            if n_matches == 0:
                dif = (pos_controls_4['AAS'] - age) / -1 # calc difference between case and all pos_controls_4
                ix = dif.idxmin() # get index of smalleset difference
                match = pos_controls_4.iloc[ix,:] # slice for the first match (if > 1)
            else:
                ix = np.random.randint(0,n_matches) # choose one randomly
                match = match.iloc[ix,:]

            matches.append(match)
            # drop patient from control pool so not re-used:
            pos_controls_4 = pos_controls_4[pos_controls_4['ID.int32()'] != match['ID.int32()']]
            pos_controls_4 = pos_controls_4.reset_index(drop=True)

    match_cons_mini = pd.concat(matches, axis=1).T.reset_index(drop=True) # store all matches
    
    plt.figure(figsize=(10,6))
    plt.title('Age dsitributions of test cases and n:{} matched controls'.format(n_controls))
    plt.xlabel('age')
    plt.ylabel('density')
    plt.hist([cases.AAS, match_cons_mini.AAS], 
             density=True,
             bins=50) #visualise distribution
    plt.legend(['cases', 'controls'])
    name = 'Test_cases_and_{}_controls_age'.format(cases.shape[0]*n_controls) + time.strftime("_%d_%b_%Y_%H%M") + '.png'
    if folder not  in [None, False]:
        plt.savefig(os.path.join(folder, name))
    plt.show()
    
    # use ANs from above to slice all columns:
    mconANs = match_cons_mini.AN.values
    match_cons = pos_controls[pos_controls['AN'].isin(mconANs)]
    
    # df of randomly sorted (frac=1), mixed cases and matched controls' episodes
    case_control_df = pd.concat([cases, match_cons], axis=0, sort=False).sample(frac=1) # stack cases and age-matched controls and shuffle 
    return case_control_df, match_cons

def age_matcher_old(df1, df2, folder, control='no_Ca'):
    """
    Takes any df, treated as cases, with 'HIST_OUTCOME.string()' and AAS (age at screening) 
    column and returns df with controls. 
    For cases, most recent episode is used for matching.
    For controls, if matching to "normal" match is two second last episode. 
    Saves:
        1. Just matched controls df
        2. Shuffled case_control_df 
            to 'folder'.
    Args:
        - df1 = cases
        - df2 = all potential controls
        - folder = dir
        - control = patient group to use as controls for age-matching eg no cancer, false_positives
    """
    print('\nAge-matching cases and controls...')
    # Make sure cases are cancer only:
    cases = df1[df1['HIST_OUTCOME.string()'] == 1]
    
    controls = df2[df2['Eps_N_to_last'] == -2] # match to most recent episode (which isn't last episode)
    # TO DO - include earlier episodes if there is one prior (above)
    
    # Sort cases and controls in ascending Age At Screening (AAS)
    controls = controls.sort_values('AAS').reset_index(drop=False)
    cases = cases.sort_values('AAS').reset_index(drop=False)
    
    del df1, df2
    gc.collect() # save some RAM
    
    matches = [] 
    for i, age in enumerate(tqdm(cases['AAS'], desc='Age-matching')):
        match = controls[controls['AAS'] == age] # slice all controls with same age
        n_matches = match.shape[0] # store n of matches
        if n_matches == 0:
            print('\nNo exact match found, searching closest aged control...')
            dif = (controls['AAS'] - age) / -1 # calc difference between case and all controls
            ix = dif.idxmin() # get index of smalleset difference
            match = controls.iloc[ix,:] # slice for the first match (if > 1)
        else:
            ix = np.random.randint(0,n_matches) # choose one randomly
            match = match.iloc[ix,:]

        matches.append(match)
        controls = controls.drop(match.name) # drop match from control pool so not re-used
        controls = controls.reset_index(drop=True)
    match_cons = pd.concat(matches, axis=1).T.reset_index(drop=True) # store all matches

    cols = match_cons.columns.tolist()
    cols.remove('AAS')
    for i in cols:
        match_cons[i] = match_cons[i].apply(lambda x: int(x))
    # save df of just matched episodes (sorted in ascending Age At Screening (AAS))
    match_cons.to_csv(folder + 'age_matched_controls.csv', index=False)
    
    plt.figure(figsize=(10,6))
    plt.title('Age dsitributions of cases and matched controls')
    plt.xlabel('age')
    plt.ylabel('counts')
    plt.hist([cases.AAS, match_cons.AAS], bins=50) #visualise distribution
    plt.legend(['cases', 'controls'])
    plt.show()
    # df of randomly sorted (frac=1), mixed cases and matched controls' episodes
    case_control_df = pd.concat([cases, match_cons], axis=0).sample(frac=1) # stack cases and age-matched controls and shuffle 
    case_control_df.to_csv(folder + 'age_matched_case_control_df.csv', index=False)
    # IDs = df['ID.int32()'].values
    return case_control_df

def applyLUT_and_window(dcm, maxv=65535, window='Normal'):
    """ 
    implemented from https://dicom.innolitics.com/ciods/digital-mammography-x-ray-image/voi-lut/00281056 
    
    # maxv determines max value and histogram of output image. 
    Must be 65535 (max possible value for np.uint16 / 16bit values)
    for saving to png with PIL / Image.fromarray()
        otherwise ouput is way too dark.
    
    Takes decompressed mammogram dicom and applies windowing
    ::Args::
        ::image - dicom pixel array
        :: dcm - pydicom object type : dataset.FileDataset (dcm metadata)
        :: window as per mammograms, default = 'Normal'
            - 'User'
            - 'Bright'
            - 'Normal'
            - 'Dark'
        :: maxv
            -  65535 for np.int16 max value and for torch transforms compatability
            - # based on dcm.BitsStored

    """
    try:
        assert dcm.PresentationLUTShape == 'INVERSE', dcm.PhotometricInterpretation == 'MONOCHROME1'
    except:
        AttributeError, print('\ndcm.AccessionNumer:', dcm.AccessionNumber)
    image = dcm.pixel_array.astype(float)
    
    print(dcm.Manufacturer)

    # store window centres and widths:

    win = pd.DataFrame([list(dcm.WindowCenter), list(dcm.WindowWidth)], index=['dcm.WindowCenter', 'dcm.WindowWidth'], columns=list(dcm.WindowCenterWidthExplanation))
    WC = int(np.round(win.loc['dcm.WindowCenter',window]))
    WW = int(np.round(win.loc['dcm.WindowWidth',window]))
    
    # apply sigmoid LUT function:    
    image = (4 * ((image - WC) / WW)) # positive 4 (versus -4) inverts
    
    image = maxv / (1 + (np.exp(image)))
    
    return image.astype(np.float32) #.astype(np.int16)

def applyLUT_and_window_exp(dcm, maxv=32767, window='Normal'):
    """ 
    implemented from https://dicom.innolitics.com/ciods/digital-mammography-x-ray-image/voi-lut/00281056 
    """
    assert dcm.PresentationLUTShape == 'INVERSE', dcm.PhotometricInterpretation == 'MONOCHROME1'
    image = dcm.pixel_array.astype(np.int16) #
    # store window centres and widths:
    win = pd.DataFrame([list(dcm.WindowCenter), list(dcm.WindowWidth)], index=['dcm.WindowCenter', 'dcm.WindowWidth'], columns=list(dcm.WindowCenterWidthExplanation))
    WC = win.loc['dcm.WindowCenter',window]
    WW = win.loc['dcm.WindowWidth',window]
    # apply sigmoid LUT function:    
    image = (4 * ((image - WC) / WW)) # positive 4  (-4 remains inverted)
    image = maxv / (1 + (np.exp(image)))
    return image.astype(np.int16)

def applyLUT_and_window_exp2(dcm, maxv=65535, window='Normal'):
    """ 
    implemented from https://dicom.innolitics.com/ciods/digital-mammography-x-ray-image/voi-lut/00281056 
    """
    assert dcm.PresentationLUTShape == 'INVERSE', dcm.PhotometricInterpretation == 'MONOCHROME1'
    image = dcm.pixel_array.astype(np.int16) #
    # store window centres and widths:
    win = pd.DataFrame([list(dcm.WindowCenter), list(dcm.WindowWidth)], index=['dcm.WindowCenter', 'dcm.WindowWidth'], columns=list(dcm.WindowCenterWidthExplanation))
    WC = win.loc['dcm.WindowCenter',window]
    WW = win.loc['dcm.WindowWidth',window]
    # apply sigmoid LUT function:    
    image = (4 * ((image - WC) / WW)) # positive 4  (-4 remains inverted)
    numer = dcm.LargestImagePixelValue - dcm.SmallestImagePixelValue
    image = numer / (1 + (np.exp(image)))
    image -= image.min()
    print(image.min(), image.max())
    return image.astype(np.uint16)

def build_train_val_test_hdfs(cio, master_list, hdf_dir, folder, train_name, val_name, test_name):
    # class instance object (cio) from BSSA_dataset_build
    #build train set:
    train_IDs, other_IDs = train_test_split(master_list,
                                        train_size=500,
                                        test_size=200,
                                        shuffle=False)
    val_IDs, test_IDs = train_test_split(other_IDs,
                                     train_size=0.5,
                                     test_size=0.5,
                                     shuffle=False)   
    
    cio.build_hdf(train_IDs, h5name=train_name,  
       out_folder=folder,
       scaleto=(250,250)) # 'Wu_NYU_2019')    
    
    cio.build_hdf(val_IDs, h5name=val_name,  
       out_folder=folder,
       scaleto=(250,250))
    
    cio.build_hdf(test_IDs, h5name=test_name,  
       out_folder=folder,
       scaleto=(250,250))
    
    items_list = cio.m_list
    
    return

def Ca_annots():
    """ on box, returns df of cancer patients with annotations """
    
    df = pd.read_csv('/home/mlim-user/Documents/james/tempdir/dcm_master_info_v2.csv')
    df['annots'] = df['annots'].astype(str)
    
    df = df[df['annots'] == '1']
    df.reset_index(drop=True, inplace=True)
    
    df2 = pd.read_csv('/home/mlim-user/Documents/james/tempdir/coded_megadf.csv')
    
    df2 = df2[df2['HIST_OUTCOME.string()'] == 1]
    
    Ca_IDs = df2['ID.int32()'].values
    Ca_IDs = ['I' + str(x) for x in Ca_IDs]
    ca_annots = df[df['ID'].isin(Ca_IDs)]
    del Ca_IDs, df, df2
    gc.collect()
    return ca_annots

#def calc_m_v(hdf_dir, train_h5IDs, h5_name):
#    """ Uses pytorch dataloader and custom dataset to calculate dataset, channel-wise mean and variance
#    ::Args::
#        :: h5_path eg '/home/mlim-user/Documents/james/h5s/'
#        :: train_h5IDs in list format eg:
#                - ['Ca/I764193-E4-A700395-S/RCC',
#                     'No_ca/I750486-E4-A645400-S/LCC',
#                     'No_ca/I645222-E6-A445206-S/LCC']
#        :: h5_name eg 'BSSA_train_v4.h5',
#    Returns 2 floats (mean and std)
#    """
#    pre_set = bs_loader(list_IDs=train_h5IDs,
#                                hdf_file=hdf_dir+h5_name,
#                                transform=None, 
#                                oversampling=None, 
#                                vis_input=None)
#    loader = DataLoader(
#        pre_set,
#        batch_size=1,
#        num_workers=1,
#        shuffle=False
#    )
#    
#    #train_mean, train_std = 3431.7081, 798.0642
#    train_mean, train_std = online_mean_and_sd(loader)
#    return train_mean.item(), train_std.item()
    
def coldict():

    #function to make a dictionary of col number : col code
    # add col description later
    # Sample (for crap RAM / whole csv hangs)
    df = pd.read_csv(csvdir + '/combined_extract.csv', date_parser=True, delimiter='\t', header=None, 
                   infer_datetime_format=True, low_memory=True, 
                   names=(pd.read_csv(csvdir + '/header.csv', header=None)[0]), 
                   nrows=3, usecols=None)

    cols = []
    for i in range(len(df.columns)):
        cols.append(df.columns[i])
    
    colix = list(range(len(cols)))

    coldict = dict(zip(colix, cols))
    return coldict


coldict = coldict()

def CaIDs(csvdir):
    ''' 
    Input dir of combined_extract.csv
    Return array of unique pt IDs: 
    '''
    
    # load df of IDs and GT ('HIST_OUTCOME.string()')
    df = pd.read_csv(csvdir + 'combined_extract.csv', delimiter='\t', header=None, 
                 names=(pd.read_csv(csvdir + 'header.csv', header=None)[0]),
                 usecols=['ID.int32()','HIST_OUTCOME.string()'])
    
    df = clean_IDs(df)
   
    CaIDs = df[df['HIST_OUTCOME.string()'] == 'Malignant Breast Cancer - 1']['ID.int32()'].values
    CaIDs.sort(axis=0)
    # Double check that IDs have cancer in 'HIST_OUTCOME.string()' column:
    for ID in CaIDs:
        assert 'Malignant Breast Cancer - 1' in df[df['ID.int32()'] == ID]['HIST_OUTCOME.string()'].values, "'Malignant Breast Cancer - 1' not associated with that ID. "
    
    return CaIDs


def CaFreeIDlist():
    """ create array of patients with at least 2 episodes and no Assessments
    (as opposed to women who were recalled, had biopsies and surgery and histological 
    evidence of non-malignant or pre-malignant abnormalities) """
    arr = np.load('/home/james/Documents/BSSA-loc/jc/megarray3.npy')
    # Slice for women with at least 2 episodes.
    arr = arr[arr[:,-3]>=2,:]
    # Find IDs of women who had any HIST_OUTCOME, other than nan:
    NotNaNRowIX = np.where(arr[:,402] != 2)
    NotNaN_IDs = arr[NotNaNRowIX, 0]
    # remove those women
    mask = ~np.isin(arr[:,0], NotNaN_IDs) # bool mask for ID col values in list of women who ever had histology
    arr = arr[mask,:]    
    IDs = np.unique(arr[:,0])
    np.save(csvloc + 'jc/NoHist_>=2eps.npy', IDs)
    return IDs

def check_files_on_disk(cropped_im_dir, df):
     
    files = sorted(os.listdir(cropped_im_dir))
    ondiskANs = [x.split('-',3)[2] for x in files]
    not_on_disk = 0
    for AN in df['AN']:
        if AN in ondiskANs:
            pass
        else:
            print("{} not on disk (".format(AN)+cropped_im_dir+')')
            not_on_disk += 1
    if not_on_disk == 0:
        print('\nAll patients have at least one hdf5 in', cropped_im_dir)
    else:
        print(not_on_disk, "\npatients' ANs not found in ", cropped_im_dir)
    
def check_h5_is_ready_sep(fp, dict_fp, hm=False):
    """ Input separate h5 parent dir, scans for image and attributes 
    Pre training dry run to assert image and attrs don't raise errors.
    Also a pre-training check that dictionary matches directory """
    files = [os.path.join(fp, x) for x in os.listdir(fp)]
    exam_list = pickling.unpickle_from_file(dict_fp)
    count = 0
    for exam in exam_list:
        for view in VIEWS.LIST:
            name = exam[view][0]
            filep = os.path.join(fp, name + '.hdf5')
            assert filep in files
        
    for file in tqdm(files):
        count += 1
        f = h5py.File(file, 'r')
        if hm:
            name = 'hm_image'
        else:
            name = 'image'
        im = f[name][()]
        assert im is not None
        lca = f[name].attrs['Left_views_ca']
        lb = f[name].attrs['Left_views_benign']
        rca = f[name].attrs['Right_views_ca']
        rb = f[name].attrs['Right_views_benign']

        f.close()
    return "Checked {} images".format(count)

def check_list(exam_list_path):
    """
    Checks a  for valid strings for filenames
    """
    exam_list = unpickle_from_file(exam_list_path)
    
    for i in range(len(exam_list)):
        for string, listyboi in exam_list[i].items():
            if isinstance(listyboi, list):
                print(listyboi)
                for file in listyboi:
                    assert isinstance(file, str)
    
    return print('\nAll filenames in', os.path.split(exam_list_path)[-1], '\nare strings.\n')

def check_multi_ims_per_view(exam_list):
    for exam in exam_list:
        for view in VIEWS.LIST:
            print('n', view, ':', len(exam[view]))
            if len(exam[view]) > 1:
                print('\n', exam)

def check_path(path):
    if path not in sys.path:
        sys.path.append(path)
        
def check_years():
    df, codes = load_df(coded=True)
    ca_pcs = []
    results = []
    for year in np.arange(2010, 2017).astype(str):
        ydf = df[df['SX_DATE.string()'].str.startswith(year)] # year df
        ca_eps = ydf[ydf['HIST_SNOMED.string()'].isin(ca_codes)]
        ben_eps = ydf[ydf['HIST_SNOMED.string()'].isin(benign_codes)]
        normies = ydf[(ydf['cancer_ep'] == 0) & (ydf['benign_tumour_ep'] == 0)]
        pc_eps_ca = ca_eps.shape[0] / normies.shape[0] # % of episodes that are cancer
        ca_pcs.append(pc_eps_ca)
        results.append(
                {
                        'year':year, 
                        'n_cancer_eps':ca_eps.shape[0], 
                        'n_bx_benign_eps':ben_eps.shape[0], 
                        'n_normal_eps':normies.shape[0],
                        '%_eps_ca':pc_eps_ca
                        
                        })
    print('mean ca_eps_pc:', np.mean(ca_pcs))
    return results

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def clean_IDs(df):
    """ Takes a df including 'ID.int32()' and cleans special characters from column"""
    print('Cleaning special characters from IDs...')
    newIDs = []
    df['ID.int32()'] = df['ID.int32()'].astype(str)
    for x in tqdm(df['ID.int32()'], desc='cleaning IDs'):
        if x.startswith('?'):
            newIDs.append(x[1:])
        else:
            newIDs.append(x)
    newIDs = [int(x) for x in newIDs]
    df['ID.int32()'] = newIDs
    return df


def compare_ttsdf_and_examlist(df, exam_list):
    """ Input df eg (test_cases.csv) and exam_list (cases and controls)
    Removes episodes from df not in exam_list
    Returns df
    """
    print('\nAsserting all df ANs are in exam_list...')
    filenames = [next(iter(x[view])) for x in exam_list for view in VIEWS.LIST]
    ANs = []
    not_found = []
    for AN in df.AN.values: # for accession numbers in df
        if any(AN in filename for filename in filenames): # if AN is found in any filenames...
            print('found df.AN {} in '.format(AN) + ' exam_list')
            ANs.append(AN)
            pass
        else:
            print('{} not_found'.format(AN) + ' in exam_list')
            not_found.append(AN)

    print('\n df shape before removing ANs not in exam_list:')
    print(df.shape)
    df = df[df.AN.isin(ANs)]
    print('\n df shape after removing ANs not in exam_list:')
    print(df.shape)
    print('\nn ANs not found: ', len(not_found))
    return df
 
def count_dir(path):
    files = os.listdir(path)
    ca_ims = [x for x in files if 'p1' in x]   
    ca_ANs = np.unique([x.split('-', 3)[2] for x in ca_ims])
    no_ca_ims = [x for x in files if 'p0' in x]
    no_ca_ANs = np.unique([x.split('-', 3)[2] for x in no_ca_ims])
    print('\n\t{} unique cancer patients'.format(len(ca_ANs)))
    print('\t{} unique no cancer patients'.format(len(no_ca_ANs)))
    
def crop(dcm_or_array, pre_or_post_LUT):
    """ Crops black-space  
    Returns array and shape
    
    Uses pixel_array mean as delinieater of breast and air
    Checks column and row-wise for inclusion in mean based mask to establish slice indices.
    
    Args:
        dcm (from dicom.read_file)
        pre_or_post_LUT : 'pre' or 'post'
        
    Returns:
        arr, cropped_dims, top_ix, bottom_ix, L_ix, R_ix """
    if isinstance(dcm_or_array, pydicom.dataset.FileDataset):
        pa = dcm_or_array.pixel_array
    elif isinstance(dcm_or_array, np.ndarray):
        pa = dcm_or_array
    
    if pre_or_post_LUT == 'pre':
        mask = pa < pa.mean()
    elif pre_or_post_LUT == 'post':
        mask = pa > pa.mean()
    # indices:
    top_ix, L_ix = 0, 0
    bottom_ix, R_ix = pa.shape
    # column-wise from left, check if mask includes anything
    for i in range(mask.shape[1]):
        L_empty = True in mask[:,i]
        if L_empty == True:
            L_ix += i
            break
    # same from right
    for i in range(mask.shape[1]):
        R_empty = True in mask[:,-i-1]
        # first non-empty row is R boundary:
        if R_empty == True:
            R_ix -= i
            break
    # search top down:
    for i in range(mask.shape[0]):
        # for each column check if mask includes
        empty = True in mask[i,:]
        if empty == True:
            top_ix += i
            break

    for i in range(mask.shape[0]):

        empty = True in mask[int(i/-1)-1,:]

        if empty == True:
            bottom_ix -= i
            break 
    # add a small border
    if L_ix > 50:
        L_ix -= 50
    if top_ix > 50:
        top_ix -= 50
    R_ix += 50
    bottom_ix += 50
    if R_ix > 4915:
        R_ix = 4915
    if bottom_ix > 5355:
        bottom_ix = 5355

    arr = pa[top_ix : bottom_ix , L_ix : R_ix]
    cropped_dims = arr.shape

    return arr, cropped_dims, top_ix, bottom_ix, L_ix, R_ix
  
def data_to_df(data, return_files=False):
    """ takes data.pkl used for model dataloading
    returns df sliced for just those episodes """
    if isinstance(data, str):
        data = unpickle_from_file(data)
    ANs = np.unique([x['L-CC'][0].split('-',3)[-2] for x in data])
    df, codes = load_df()
    df = df[df['AN'].isin(ANs)]
    if return_files:
        files = list(itertools.chain.from_iterable([x[view] for view in VIEWS.LIST for x in data])) 
        return df, files
    return df 

def decode_col(codes, col):
    """ takes coded_df (as from load_df(coded=True) and column name as string
    returns dict of coded df elements as keys and corresponding original string as values """
    
    x = codes[col].value_counts(dropna=False).shape[0]
    keys = codes[col][:x].index.values
    values = codes[col][:x]
    return dict(zip(keys, values))

def dslice(cols):
    """
    # Input zero-indexed column number to load one or more cols
    # load and describe df of certain columns from csv
    # due to memory 'restrictions' (1GB csv) 
    """
    colnames = []
    for i in cols:
        colnames.append(coldict.get(i))
        
    df = pd.read_csv(csvloc + 'combined_extract.csv', date_parser=True, delimiter='\t', header=None, 
                   infer_datetime_format=True, low_memory=True, 
                   names=colnames, 
                   nrows=None, usecols=cols)
    """
    col = df.columns[0]
    
    # rename cols labelled 'string()' in error:

    if 'AX_MAMM_LES' in col:
        df = df.rename(index=str, columns={col:col[0:16] + '.int32()'})
    """ 

    for i in cols:
        print('Col#:', i, '= Column code:', coldict.get(i))
    
    
    #    print(df.describe())          
    return df 

def distal_crop(dcm):
    """ Crops distal black-space in width only 
    Returns array and shape
    
    Uses pixel_array mean as delinieater of breast and air
    Checks column-wise for inclusion in mean based mask.
    
    Args:
        dcm (from dicom.read_file)"""
    mean = dcm.pixel_array.mean()
    mask = dcm.pixel_array < mean
    # compare plt.imshow(dcm.pixel_array) and plt.imshow(mask)
    ix = 0 # start index of first non-empty column
    lat = dcm.ImageLaterality
    if lat == 'L':
        for i in range(1, mask.shape[1]):
            # for each column check if mask includes
            empty = True in mask[:,-i] # start from empty/right of image for L dcms
#            print(i, empty)
            if empty == True:
                ix += i
                break
        arr = dcm.pixel_array[:,:-ix+100]
    if lat == 'R':
        for i in range(1, mask.shape[1]):
            empty = True in mask[:,i] # start from empty/left of image for R dcms. 
            if empty == True:
                ix += i
                break
        arr = dcm.pixel_array[:,ix-100:]
    cropped_dims = arr.shape
    plt.imshow(arr)
    return arr, cropped_dims


def load_ca_pts_df_clean(csvloc, eps):
    """ slices combined extract for Cancer patients'
        iterates through chunks of full dataset (too big to load in RAM) """
    reader =  pd.read_csv(csvloc + 'coded_megadf.csv', chunksize=5000, low_memory=False)
    dflist = []
    Cancer_IDs = CaIDs(csvloc)
    for i, chunk in tqdm(enumerate(reader)):    
        if eps == 'Ca_only':
            chunk = chunk[chunk['HIST_OUTCOME.string()'] == 1.0]
        elif eps == 'all':
            chunk = chunk[chunk['ID.int32()'].isin(Cancer_IDs)]
        dflist.append(chunk)
    df = pd.concat(dflist)
    df = df.sort_values(by='ID.int32()')
    return df

def load_ca_pts_df_dirty(csvloc, eps):
    """ slices combined extract for Cancer patients'
        iterates through chunks of full dataset (too big to load in RAM) """
    reader =  pd.read_csv(csvloc + 'combined_extract.csv', chunksize=1000, date_parser=True, delimiter='\t', infer_datetime_format=True, iterator=False, low_memory=False, names=(pd.read_csv(csvloc + 'header.csv', header=None)[0]))
    dflist = []
    Cancer_IDs = CaIDs(csvloc)
    for i, chunk in tqdm(enumerate(reader)):    
        # Most IDs are ints, some have special characters
        chunk = clean_IDs(chunk)
        if eps == 'Ca_only':
            chunk = chunk[chunk['HIST_OUTCOME.string()'] == 'Malignant Breast Cancer - 1']
        elif eps == 'all':
            chunk = chunk[chunk['ID.int32()'].isin(Cancer_IDs)]
        dflist.append(chunk)
    df = pd.concat(dflist)
    df = df.sort_values(by='ID.int32()')
    return df

def load_df(coded=True):
    if coded == False:
        return pd.read_csv(os.path.join(csvloc, 'combined_extract.csv'))
    else:
        print('coded returns two frames (data, codes)')
        df = pd.read_csv(os.path.join(csvloc, 'coded_megadf.csv'))
        dropcols = [x for x in df.columns if '().' in x or 'Unnamed' in x]
        if len(dropcols) > 0:
            df.drop(dropcols, axis=1, inplace=True)
        if df['SX_ACCESSION_NUMBER.int32()'].dtype == float:
            df['AN'] = 'A' + df['SX_ACCESSION_NUMBER.int32()'].astype(int).astype(str)
        codes = pd.read_csv(os.path.join(csvloc, 'megadf_encoding.csv'))
        return df, codes
    

def make_master_list(csvdir, sftp, VMdir, one_folder):
    print("\nMaking master list with laterality and views...\n")
    files = sftp.listdir(VMdir + one_folder)
    files = [x for x in files if '?' not in x] # drop filenames containing '?'
    dm_df = pd.read_csv(csvdir + 'dcm_master_info_final.csv')
    df = dm_df[dm_df['filename'].isin(files)]
    print("in files", df.shape)
    df = df[df['views'].isin(['MLO','CC'])] 
    print("\ndf shape after slicing:", df.shape)
    filenames = df['full_fp'].values.tolist()
    filenames = [x.rsplit('images/',1)[-1] for x in filenames]
    file_lats = df['lat'].values.tolist()
    file_views = df['views'].values.tolist()
        
    m_list = [list(a) for a in zip(filenames, file_lats, file_views)]
    return m_list

def make_master_list_from_local(csvdir, fp, one_folder):
    print("\nMaking master list with laterality and views...\n")
    files = os.listdir(fp + one_folder)
    files = [x for x in files if '?' not in x] # drop filenames containing '?'
    dm_df = pd.read_csv(csvdir + 'dcm_master_info_final.csv')
    df = dm_df[dm_df['filename'].isin(files)]
    print("in files", df.shape)
    df = df[df['views'].isin(['MLO','CC'])] 
    print("\ndf shape after slicing:", df.shape)
    filenames = df['full_fp'].values.tolist()
    filenames = [x.rsplit('images/',1)[-1] for x in filenames]
    file_lats = df['lat'].values.tolist()
    file_views = df['views'].values.tolist()
        
    m_list = [list(a) for a in zip(filenames, file_lats, file_views)]
    return m_list


def link_combined_extract_and_dcm_meta(cohort, folder):
    """ Uses combined_extract and dicom metadata dfs to slice for cohorts
        Args:
            cohort = 'ca_free' or 'ca'
            folder = 'ML_01_files'
        Returns sliced dicom metadata df - dm_df
    """
    assert cohort in ['ca_free', 'ca']
    dm_df = pd.read_csv(
            os.path.join(csvdir, 'dcm_master_info_final.csv'),
                         low_memory=False
                        )
    dm_df = dm_df[dm_df.views.isin(['CC', 'MLO'])]
    dm_df['full_fp'] = [x.split('images/', 1)[-1] for x in dm_df.full_fp]
    dm_df = dm_df[dm_df.full_fp.str.startswith(folder)]
    
    df = pd.read_csv(
            os.path.join(csvdir, 'coded_megadf.csv'),
            low_memory=False
            )
    # create col with syntax 'A123456' to match with dm_df:
    df['AN'] = 'A' + df['SX_ACCESSION_NUMBER.int32()'].astype(int).astype(str)
    
    ca_IDs = df[df['HIST_OUTCOME.string()'] == 1]['ID.int32()'].values

    if cohort == 'ca_free':
        # use only women with at least one follow-up
        # don't use their last episode
        # make sure there are no cancer patients
        df = df[df.Total_Eps > 1]
        df = df[df.Eps_N_to_last < -1]          
        df = df[~df['ID.int32()'].isin(ca_IDs)]
        
    elif cohort == 'ca':
        df = df[df['HIST_OUTCOME.string()'] == 1.0] # cancer episode only

    else:
        raise ValueError
        
    ANs = df['AN'].values   
    dm_df = dm_df[dm_df.AN.isin(ANs)]
    del df
    gc.collect()
    return dm_df

def link_coded_megadf_and_dm_meta(dm_df, df):
    """ Takes dicom metadata df (cols = filename, ID, Ep, rsync fp ...)
    and coded df and joins. Enables matching of individual dicom files with patient data """
    dm_df.sort_values(by='AN', inplace=True)
    if df['SX_ACCESSION_NUMBER.int32()'].dtype == float:
        df['AN'] = 'A' + df['SX_ACCESSION_NUMBER.int32()'].astype(int).astype(str)
    df.sort_values(by='AN', inplace=True)
    new_dm_df = pd.merge(dm_df, df, on='AN', how='inner')
    return new_dm_df

  
def make_dm_df_im_labels(dm_df, df):
    """ Uses dicom metadata dataframe and coded_megadf.csv to correlate image-wise ca labels
    Returns list of ints
    """
    im_labels = []
    for i in tqdm(range(len(dm_df))):
        AN = dm_df.AN.values[i]
        if dm_df.lat.values[i] == 'L': # if this image is left view:
            im_labels.append(df[df.AN == AN]['Left_views_ca'].values[0]) # append the value for this episode
        elif dm_df.lat.values[i] == 'R':
            im_labels.append(df[df.AN == AN]['Right_views_ca'].values[0])
    return im_labels

def make_txt_list(cohort, save_folder, slice_folder=False, all_folders=False):
    """ uses link_combined_extract_and_dcm_meta then saves as text file
    Args:
            cohort = 'ca_free' or 'ca'
            slice_folder = 'ML_01_files'
            save_folder = directory for saving .txt files
            if all_folders=True, will call for whole ds / all folderss
    """
    if all_folders == True:
        for slice_folder in tqdm(['ML_01_files', 'ML_02_files', 'ML_03_files', 'ML_04_files', 'ML_05_files', 
                             'ML_06_files', 'ML_07-10_files', 'ML_11-13_files', 'ML_14-16_files', 
                             'ML_17-19_files', 'ML_20-22_files', 'ML_23-25_files', 'ML_26-28_files', 
                             'ML_29-31_files', 'ML_32-34_files', 'ML_35-37_files', 
                             'ML_38-40_files', 'ML_41-43_files', 'ML_44-45_files', 'ML_46_files'], desc='Slicing each folder for {}'.format(cohort) + ' patients, saving list as txt file at ' + save_folder):
            print(slice_folder)

            fp = os.path.join(save_folder, slice_folder) + '.txt'
            df = link_combined_extract_and_dcm_meta(
                    cohort=cohort,
                    folder=slice_folder)
            
            np.savetxt(
                    fname=fp,
                    X=list(df.filename),
                    fmt='%s'
                    )
    else:
        fp = os.path.join(save_folder, slice_folder) + '.txt'
        df = link_combined_extract_and_dcm_meta(
                cohort=cohort,
                folder=slice_folder)
        
        np.savetxt(
                fname=fp,
                X=list(df.filename),
                fmt='%s'
                )
        return df
    

def nyu_exam_list(im_folder, dic_fp, file_format='.dcm', ca_only=False):
    """
    Converts im_folder containing .png or .dcm mammograms to nyu-style exam-list
    [
    {'L-CC': ['0_L_CC'], #.png
  'L-MLO': ['0_L_MLO'],
  'R-CC': ['0_R_CC'],
  'R-MLO': ['0_R_MLO'],
  'horizontal_flip': 'NO'},
 {'L-CC': ['1_L_CC'],
  'L-MLO': ['1_L_MLO'],
  'R-CC': ['1_R_CC'],
  'R-MLO': ['1_R_MLO'],
  'horizontal_flip': 'NO'}
 ]
    """
    assert file_format.startswith('.')
    assert isinstance(file_format, str)
    assert file_format in ['.png', '.dcm', '.hdf5']
    assert not dic_fp.endswith('.pkl')
    out_list = []
    AN_list = []
    fails = []
    files = os.listdir(im_folder)
    files = [x for x in files if x.endswith(file_format)]
    if ca_only: # slice just for cancer patients
        files = [x for x in files if 'p1' in x]
    assert len(files) != 0
    for file in files:
        ID, ep, AN, _ = file.split('-',3)
        if AN not in AN_list:
            AN_list.append(AN)   
    
    for i, AN in enumerate(tqdm(AN_list)):
        try:
            LCC = sorted([x for x in files if AN in x and 'L-CC' in x]) #[-1]
            LMLO = sorted([x for x in files if AN in x and 'L-MLO' in x]) #[-1]
            RCC = sorted([x for x in files if AN in x and 'R-CC' in x]) #[-1]
            RMLO = sorted([x for x in files if AN in x and 'R-MLO' in x]) #[-1]
            
            for im_list in [LCC, LMLO, RCC, RMLO]:
                assert len(im_list) > 0
            
            dic = dict(
                                {'L-CC': [x.split('.',1)[0] for x in LCC],
                                 'L-MLO': [x.split('.',1)[0] for x in LMLO],
                                 'R-CC': [x.split('.',1)[0] for x in RCC],
                                 'R-MLO': [x.split('.',1)[0] for x in RMLO],
                                 'horizontal_flip': 'NO'}
                                )
            out_list.append(dic)
            
        except:
            print('view lists failed : ( for {}'.format(AN))
            fails.append(AN)
            continue
    out_fp = dic_fp + '_exam_list.pkl'
    fails_fp = dic_fp + '_fails.pkl'
    pickle_to_file(out_fp, out_list)
    pickle_to_file(fails_fp, fails)
    print('\n', out_fp, ' saved.')
    return

def onept_CE(ID, T=False):
    """
    CE == from original combined_extract.csv
    Function to input pt ID number as string and output columns of episodes
    (for easier vis of one patients' journey)
    """
    pt = sliceofall_dirty('ID.int32()', ID)
    if T == True:
        pt = pt.T
    return pt

def onept_CM(ID, csv='/home/mlim-user/Documents/james/tempdir/coded_megadf.csv', T=False):
    """
    CM == from coded_megadf.csv
    Function to input pt ID number as string and output columns of episodes
    (for easier vis of one patients' journey)
    """
    df = pd.read_csv(csv)
    pt = df[df['ID.int32()'] == ID]
    if T == True:
        pt = pt.T
    return pt

def online_mean_and_sd(loader):
    """ https://discuss.pytorch.org/u/xwkuang5/summary
    Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(1, dtype=torch.float64)
    snd_moment = torch.empty(1, dtype=torch.float64)

    for _,data,_,_ in tqdm(loader):

        c, h, w = data.shape
        nb_pixels = h * w
        sum_ = torch.sum(data, dim=[0, 1, 2])
        sum_of_square = torch.sum(data ** 2, dim=[0, 1, 2])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

def pickle_to_file(file_name, data, protocol = pickle.HIGHEST_PROTOCOL):
    # https://stackoverflow.com/questions/36745577/how-do-you-create-in-python-a-file-with-permissions-other-users-can-write
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol)

def plot_learning(dic):
    """ Args:
            hard coded dictionary from box_main_v6 """
    plt.clf()
    plt.plot(dic['train_acc'])
    plt.plot(dic['val_acc'])
    plt.legend(['train_accuracy', 'val_accuracy'])
    plt.show()
    plt.clf()
    
    plt.plot(dic['train_loss'])
    plt.plot(dic['val_loss'])
    plt.legend(['epochs_mean_train_loss',
                'epochs_mean_val_loss',
                ])
    plt.show()
    
    plt.clf()
    plt.plot(dic['val_AUC'])
    plt.legend(['val_AUCs'])
    plt.show()
    plt.clf()
    
def plot_learning_grid(folder, dic, saveas='/res'):
    plt.clf()
    fig, axes = plt.subplots(nrows=2,ncols=2, sharex='all', figsize=(10,10))
    fig.tight_layout()
    fig.subplots_adjust(top=0.89)
    axes[0,0].plot(dic['train_acc'])
    axes[0,0].plot(dic['val_acc'])
    axes[0,0].legend(['train_accuracy', 'val_accuracy'], loc='upper left') #, bbox_to_anchor=(0.01,1.08))
    
    axes[0,1].plot(dic['train_loss'])
    axes[0,1].plot(dic['val_loss'])
    axes[0,1].legend(['mean_train_loss',
                'mean_val_loss',
                ], loc='upper left') #, bbox_to_anchor=(0.01,1.08))
    
    axes[1,0].plot(dic['val_AUC'])
    axes[1,0].legend(['val_AUCs'], loc='upper left') #, bbox_to_anchor=(0.01,1.08))
    
    title = dic['title'].replace(' ','\n')
    details = title.replace('/home/mlim-user/Documents/james/mamdl/models/training/','~')
    axes[1,1].text(0,0.5,details)
    
    title = dic['title']
    fig.suptitle(title, y=0.98, fontsize=16)
    fig.savefig(folder + saveas + '.png')
    plt.show()
    
def printname(name):
    """ helper for viewing hdf structures """
    return name

def rename_dcm_dir(dm_df_dir, local_folder, split, cohort):
    """
    Renames dicom dir from 
    /data/james/BSSA_images/train/I99993-E13-A2926127-S-i5.dcm
    to
    /data/james/BSSA_images/train/I99993-E13-A2926127-S-i5_L_MLO
    _p0i0.dcm
    """
    assert cohort in ['cases', 'controls']
    if split is not None:
        assert split in ['train', 'val', 'test']
        dm_df = unpickle_from_file(os.path.join(dm_df_dir, split+'_'+cohort+'_dm_df.pkl'))
    if 'dev' in local_folder:
            dm_df = unpickle_from_file(os.path.join(dm_df_dir, 'dev_'+cohort+'_dm_df.pkl'))
            print('\tdev dicom metadata df version loaded in utilities.all_utils.rename_dcm_dir()')
    else:
        dm_df = unpickle_from_file(os.path.join(dm_df_dir, cohort+'_dm_df.pkl'))
        print('\tfull dicom metatdata df unpickled in utilities.all_utils.rename_dcm_dir()')
        print('\t(not mini/dev version')
    
    df = pd.read_csv(
            os.path.join(csvdir, 'coded_megadf.csv'),
            low_memory=True,
            usecols=basecols
            )
    df['AN'] = 'A' + df['SX_ACCESSION_NUMBER.int32()'].astype(int).astype(str)
    # add column for patient-wise and image-wise labels:
    if cohort == 'ca':
        dm_df['pt_label'] = 1
        dm_df['im_label'] = make_dm_df_im_labels(dm_df, df)
    elif cohort == 'ca_free':
        dm_df['pt_label'] = 0
        dm_df['im_label'] = 0

    label_code = 'p' + dm_df['pt_label'].astype(str) + 'i' + dm_df['im_label'].astype(str)
    new_names = dm_df.filename.str.replace('.dcm', '') + '_' + \
        dm_df['lat'] + '_' + dm_df.views + '_' + label_code + '.dcm'
    dm_df['new_filename'] = new_names
    
    for i in tqdm(range(len(dm_df)), desc='Renaming files'):
        try: # the original filename (I99993-E13-A2926127-S-i5.dcm)
            file = dm_df.filename.values[i]
            new_name = dm_df.new_filename.values[i]
            if not os.path.exists(os.path.join(local_folder, new_name)):
                os.rename(
                        src=os.path.join(local_folder, file),
                        dst=os.path.join(local_folder, new_name)
                        )
        except:
            print(dm_df.filename.values[i], ' failed. Skipped')
            continue
        
def rename_dcm_dir_v2(slice_folder, local_folder, cohort):
    """
    Renames dicom dir from 
    /data/james/BSSA_dcm_files/ML_01_files/I99993-E13-A2926127-S-i5.dcm
    to
    /data/james/BSSA_dcm_files/ML_01_files/I99993-E13-A2926127-S-i5_L_MLO
    _p0i0.dcm
    """
    dm_df = link_combined_extract_and_dcm_meta(
            cohort=cohort,
            folder=slice_folder)
    
    dm_df['ID'] = dm_df.ID.str.replace('I','').astype(int)
    
    df = pd.read_csv(
            os.path.join(csvdir, 'coded_megadf.csv'),
            low_memory=False
            )
    df['AN'] = 'A' + df['SX_ACCESSION_NUMBER.int32()'].astype(int).astype(str)
    # pt-wise labels
    if cohort == 'ca':
        dm_df['pt_label'] = 1
    elif cohort == 'ca_free':
        dm_df['pt_label'] = 0
    # add column for image-wise label:
    dm_df['im_label'] = make_dm_df_im_labels(dm_df, df)

    label_code = 'p' + dm_df['pt_label'].astype(str) + 'i' + dm_df['im_label'].astype(str)
    new_names = dm_df.filename.str.replace('.dcm', '') + '_' + \
    dm_df['lat'] + '_' + dm_df.views + '_' + label_code + '.dcm'
    dm_df['new_filename'] = new_names
    
    ANs = df['AN'].values   
    dm_df = dm_df[dm_df.AN.isin(ANs)]
    
    for i in tqdm(range(len(dm_df)), desc='Renaming files'):
        try: # the original filename (I99993-E13-A2926127-S-i5.dcm)
            file = dm_df.filename.values[i]
            new_name = dm_df.new_filename.values[i]

            if not os.path.exists(os.path.join(local_folder, new_name)):
                os.rename(
                        src=os.path.join(local_folder, file),
                        dst=os.path.join(local_folder, new_name)
                        )
        except: # this one (I99993-E13-A2926127-S-i5_L_MLO_p0i0.dcm)
            file = dm_df.filename.values[i].replace('.dcm', '') + '_' + dm_df.lat.values[i] + '_' + dm_df.views.values[i] + '.dcm'
            if not os.path.exists(os.path.join(local_folder, new_name)):
                os.rename(
                        src=os.path.join(local_folder, file),
                        dst=os.path.join(local_folder, new_name)
                        )

def revert_codes(df, col):
    """ takes coded df and adds col in original classes and elements found in combined_extract.csv"""
    cedf = load_df(coded=False)[['SX_ACCESSION_NUMBER.int32()',col]]
    cedf.rename(columns={col:col+'_old'}, inplace=True)
    cedf['AN'] = cedf['SX_ACCESSION_NUMBER.int32()'].apply(lambda x: 'A'+str(x))
    return pd.merge(df, cedf, on='AN')
    
def clean_dcm_dir(slice_folder, local_folder, cohort):
    """
    ***** #####  
    Deletes dicom from local disk 
    ***** ####
    if not in specified folder and cohort
    """
    dm_df = link_combined_extract_and_dcm_meta(
            cohort=cohort,
            folder=slice_folder)
    
    files = os.listdir(local_folder)
    files = [x for x in files if x.endswith('.dcm')]
    files = [x for x in files if '-S-' in x]
    new_names = dm_df.filename.str.replace('.dcm', '') + '_' + \
    dm_df['lat'] + '_' + dm_df.views + '.dcm'
    dm_df['new_filename'] = new_names
    new_files = dm_df.new_filename.values
    count1 = 0    
    count2 = 0
    dcms = [x for x in os.listdir(local_folder) if x.endswith('.dcm')]
    for file in tqdm(dcms, desc='Deleting unwanted files (non-MLO, non-CC, non-screening)'):
        if file not in new_files:
            count1 += 1
            os.remove(os.path.join(local_folder, file))

        if 'CC' not in file and 'MLO' not in file and os.path.exists(os.path.join(local_folder, file)):
            count2 += 1
            os.remove(os.path.join(local_folder, file))
            
    return print('Finished cleaning', slice_folder, '\nTotal deleted:', count1+count2)


def save_dic_of_lists(folder, dic_of_lists):
    """ NOT WORKING """
    if not os.path.exists(folder):
        os.mkdir(folder)
    for i in dic_of_lists:
        pickle_to_file(folder+str(i), data=dic_of_lists)
    return print(str(i), 'saved to', folder+str(i))

def sens_spec_matrix_df(model_name, res, float):
    """
    Take res model name:
        ['NYU1_naive', 'NYU1_scratch', 'NYU1_tl', 'NYU2_naive', 'NYU2_scratch', 'NYU2_tl', 'NYU2_tl_13pc']
    """
    sens_i = res[model_name]['tpr']
    spec_i = 1 - res[model_name]['fpr']
    
    df = pd.DataFrame([sens_i, spec_i]).T.rename(columns={0:'sens',1:'spec'})
    
    tn, fp, fn, tp = res[model_name]['CM'].ravel()
    
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    acc = (tp + tn) / (tp + tn + fn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    print('\nModel: ', model_name)
    print('sens:', sens)
    print('spec:', spec)
    print('acc:', acc)
    print('ppv:', ppv)
    print('npv:', npv)
    print('n fps: ', fp)
    print('n tps: ', tp)

    return df[df['sens'] >= float]


def sliceofall_dirty(column, element):
    """
    Function to slice whole combined_extract (435 000 x 471) per a condition:
        - 'column' (input) == element (input).
    Avoids loading whole combined_extract.csv due to hardware limitations.
    """
    # iterates through chunks of full dataset (if too big to load in RAM)
    # load whole df reader 
    reader =  pd.read_csv(os.path.join(csvdir, 'combined_extract.csv'), chunksize=5000, 
                          date_parser=True, delimiter='\t', infer_datetime_format=True, 
                          iterator=False, low_memory=False, 
                          names=(pd.read_csv(
                                  os.path.join(csvloc, 'header.csv'), header=None)[0]))
    dflist = []
    for i, chunk in enumerate(reader):    
        # Most IDs are ints, some have special characters
        chunk = clean_IDs(chunk)
        chunk = chunk[chunk[column] == element]
        # saves slices into list
        dflist.append(chunk)

    df = pd.concat(dflist)
    return df


def show_2(im1, im2, title=None, stitle1=None, stitle2=None, figsize=(20,20), save=False, fp1=False, fp2=False):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharex=True, sharey=True, squeeze=True)
    fig.suptitle(title)
    #plt.figure(figsize=(15,15))
    ax[0].imshow(im1, cmap='gray')
    ax[0].title.set_text(stitle1)
    #plt.show()

    #plt.figure(figsize=(15,15))
    ax[1].imshow(im2, cmap='gray')
    ax[1].title.set_text(stitle2)

    if save:
        #panel A:
        #Save just the portion _inside_ the second axis's boundaries
        extent0 = ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(fp1, bbox_inches=extent0)
        
        # Pad the saved area by 10% in the x-direction and 20% in the y-direction
        fig.savefig(fp1 + '_expanded', bbox_inches=extent0.expanded(1.15, 0.6))
        
        # panel B:
        extent1 = ax[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(fp2, bbox_inches=extent1)
        
        # Pad the saved area by 10% in the x-direction and 20% in the y-direction
        fig.savefig(fp2 + '_expanded', bbox_inches=extent1.expanded(1.15, 1.1))
         
        # fig.savefig(fname=fp1 + '_both', dpi=600)
#    fig.tight_layout()
    plt.show()
    
def show_a(fp):
    dcm = pydicom.read_file(fp)
    plt.figure(figsize=(40,40))
    plt.imshow(dcm.pixel_array, cmap='gray')
    return

def show_heatmaps(title, im, bhm, chm, view, AN, yben, yca):
    
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(35,25), squeeze=True) #, sharex=True, sharey=True)

    ax1.imshow(im, cmap='gray')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.title.set_text(view + '\n' + AN)

    ax2.imshow(im, cmap='gray')
    ax2.imshow(bhm, cmap='jet', alpha=0.2)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.title.set_text('Benign Heatmap:\n' + yben)

    ax3.imshow(im, cmap='gray')
    ax3.imshow(chm, cmap='jet', alpha=0.2)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.title.set_text('Malignant Heatmap: \n' + yca)
    
    fig.suptitle(title, fontsize=17) 
    fig.subplots_adjust(wspace=0, hspace=0, top=1.31)
    plt.show()
        
def show_s(fp):
    dcm = pydicom.read_file(fp)
    pa = applyLUT_and_window_exp(dcm)
    plt.figure(figsize=(40,40))
    plt.imshow(pa, cmap='gray')
    return
    

def show_sep_h5s(folder, n=4, size=10, view=None):
    # visualise folder of seperate .hdf5s
    paths = [os.path.join(folder, x) for x in os.listdir(folder)]
    if view is not None:
        paths = [x for x in paths if view in x]
    for i, path in enumerate(paths):
        f = h5py.File(path, 'r')
        im = f['image'][()].T
        plt.figure(figsize=(size,size))
        plt.title(path)
        if len(im.shape) == 3:
            plt.imshow(im[:,:,0], cmap='gray')
        else:
            plt.imshow(im, cmap='gray')
        plt.show()
        f.close()
        if i == n:
            break
    return


def vis_h5_folder(folder, AN=False, limit_to=10, hm=False):
   # Iterates through folder, visualising hdf5 mammograms 
    files = [x for x in os.listdir(folder) if x.endswith('.hdf5')]
    if AN is not False:
        files = sorted([x for x in files if str(AN) in x]) 
    else:
        files = files[:limit_to]
    if hm == True:
        name = 'hm_imagev'
    else:
        name = 'image'
    for file in files:
        f = h5py.File(os.path.join(folder, file), 'r')
        try:
            plt.imshow(f[name][()].T, cmap='gray')
        except:
            plt.imshow(f[name][()][:,:,0], cmap='gray')
        plt.title(file + '\nSize:' + str(f[name][()].shape))
        plt.show()
        
    return


def colcode(obj):
    """ Decodes dictionary for columns.
    Loads dictionary, returns key for val and val for key """
    assert isinstance(obj, str) or isinstance(obj, int)
    dic = unpickle_from_file(tempdir + 'coldict')
    if type(obj) == int:
        return dic.get(obj)
    else:
        for key, val in dic.items():
            if val == obj:
                res = key
    return res

def zero1_norm(array):
    array = ((array - array.min()) / (array.max() - array.min()))
    return array

def zero_m_unit_v_norm(array):
    array -= np.mean(array)
    array /= np.std(array)
    return array

def add_imwise_labels_to_super(df):
    def lat_ca_con(lat, LVca, RVca):
        if lat == 'R':
            return RVca
        elif lat == 'L':
            return LVca
        else:
            raise SyntaxError("Laterality not Left or Right")
    def lat_benign_con(lat, LVben, RVben):
        if lat == 'R':
            return RVben
        elif lat == 'L':
            return LVben
        else:
            raise SyntaxError("Laterality not Left or Right")
    
    df['ca_im'] = [lat_ca_con(lat, LVca, RVca) for (lat, LVca, RVca) in zip(df['lat'], df['Left_views_ca'], df['Right_views_ca'])]
    df['benign_im'] = [lat_ca_con(lat, LVben, RVben) for (lat, LVben, RVben) in zip(df['lat'], df['Left_views_benign'], df['Right_views_benign'])]
    return df

def add_name_codes_to_super(df):
    df['codes'] = df['lat'] + '-' + df['views'] + '_p' + df['ca_pt'].astype(str) + 'e' + df['cancer_ep'].astype(str) + 'i' + df['ca_im'].astype(str)
    df['new_names'] = [x.split('.dcm')[0] + '-' + y + '.dcm' for (x,y) in zip(df['filename'], df['codes'])]

