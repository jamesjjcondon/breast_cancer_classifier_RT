#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:59:02 2020

@author: James Condon
"""

# This file is part of a modified version of breast_cancer_classifier:
# https://github.com/nyukat/breast_cancer_classifier
# Wu N, Phang J, Park J et al. Deep neural networks improve radiologists’ performance in breast cancer screening.
# PubMed - NCBI [Internet]. [cited 2020 Mar 6]. Available from: https://www.ncbi.nlm.nih.gov/pubmed/31603772
#
# breast_cancer_classifier is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# breast_cancer_classifier is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with breast_cancer_classifier.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, precision_recall_curve, balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import resample
from src.utilities.all_utils import load_df
from src.constants import TTSDIR, CSVDIR

from IPython import embed

#%%

def add_result_columns(df, VENDF_PATH, ca_v_norm=False):
    """ 
    1. Adds patient malignancy and benign score columns (max of left and right scores)
    2. Establishes binary outcome based on thresholds and score space
    3. Links BSSA data and ground-truth to results
    Input: inference output .csv 
    Output: merged df including BSSA clinical data and patient malignancy and benign scores
    """
    rows_in = df.shape[0]
    df['pt_ca_score'] = np.max([df['left_malignant'], df['right_malignant']], axis=0)
    df['pt_ben_score'] = np.max([df['left_benign'], df['right_benign']], axis=0)
    df['AN'] = [x.split('-',3)[2] for x in df.BSSA_ID]

    # test a range of probability cut-off thresholds for binary patient-wise predictions:
    thresholds = np.round(np.arange(0.0125, 0.45, .0125), 4) # generate a range of binary predictions for different probability thresholds
    
    for thresh in thresholds: 
        df['ca_bin_pred_thresh_'+str(thresh)] = (df['pt_ca_score'] >= thresh).apply(lambda x: int(x))

    df = df.sort_values('AN').reset_index(drop=True)

    # load and merge BSSA data for reference:
    if VENDF_PATH in [False, None]: # then load 16-8pc incidence test df
        name = 'coded_megadf.csv'
        print('loading ', name)
        df2 = pd.read_csv(
                os.path.join(
                        CSVDIR, name
                        )
                )
        df2['AN'] = ['A' + str(int(x)) for x in df2['SX_ACCESSION_NUMBER.int32()']]
    
    else:
        print('loading vendor df ', VENDF_PATH)
        df2 = pd.read_csv(
                VENDF_PATH,
                ) #[['vendor', 'AN']]
#    _, codes = load_df()
    dfm = pd.merge(df, df2, on='AN')
    
    if ca_v_norm:
        print('\n\t...excluding benign patients...')
        dfm = dfm[dfm['benign_pt'] != 1]
    else:
        print('\n\t...including benign patients...')
        
        rows_out = dfm.shape[0]
        assert rows_out == rows_in, ("Lost episodes, ?pd.merge error ?n-exams flag for inference")
    return dfm

def bootstrap_AUC(df, channels, n_resamples):
    """ Returns tuple of 95% CI """
    strapped_AUCs = []
    if channels is not None and channels is not False:
        col = 'pt_ca_score_c' + str(channels)
    elif channels is False or None:
        col = 'pt_ca_score'
    df = df[[col, 'cancer_ep']]
    print('\n\nBootstrapping {} channel results'.format(channels))
    for _ in tqdm(range(n_resamples)):
        new_df = resample(df)
        score = new_df[col]
        y = new_df['cancer_ep']
        fpr, tpr, roc_thresholds = roc_curve(y, score)
        AUC = auc(fpr, tpr)
        strapped_AUCs.append(AUC)
    ordAUCs = sorted(strapped_AUCs)
    lci = np.round(ordAUCs[int(0.025 * len(ordAUCs))], 3)
    uci = np.round(ordAUCs[int(0.975 * len(ordAUCs))], 3)
    return (lci, uci)
    
def bootstrap_PRAUC(df, channels, n_resamples):
    strapped_PRAUCs = []
    if channels is not None and channels is not False:
        col = 'pt_ca_score_c' + str(channels)
    elif channels is False or None:
        col = 'pt_ca_score'
    df = df[[col, 'cancer_ep']]
    print('\n\nBootstrapping {} channel results'.format(channels))
    for _ in tqdm(range(n_resamples)):
        new_df = resample(df)
        score = new_df[col]
        y = new_df['cancer_ep']
        ppv, sens, pr_thresholds = precision_recall_curve(y, score)
        PRAUC = auc(sens, ppv)
        strapped_PRAUCs.append(PRAUC)
    ordAUCs = sorted(strapped_PRAUCs)
    lci = np.round(ordAUCs[int(0.025 * len(ordAUCs))], 3)
    uci = np.round(ordAUCs[int(0.975 * len(ordAUCs))], 3)
    return (lci, uci)

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

def _metrics(true, pred):
    cm = confusion_matrix(true, pred)
    tn, fp, fn, tp = cm.ravel()
    ppv, sens, f1, _ = precision_recall_fscore_support(
            true, pred, beta=1.0, average='binary') # beta is balance between prec and recall (1.0 =)
    _, _, f2, _ = precision_recall_fscore_support(true, pred, beta=.5, average='binary')
    _, _, f3, _ = precision_recall_fscore_support(true, pred, beta=.3, average='binary')
    acc = (tn + tp) / cm.sum()
    spec = tn / (tn + fp)
    sens, spec, ppv, f1, f2, f3, acc = np.round([sens, spec, ppv, f1, f2, f3, acc], 3)
    npv = tn / (tn + fn)
#    print('\nPrecision/PPV:', prec, '\nRecall/Sens:', recall, '\nF1:', f1)
#    print('\nAccuracy:', acc, '\nSpecificity:', spec)
    print('sens:', sens)
    print('spec:', spec)
    print('acc:', acc)
    print('ppv:', ppv)
    print('npv:', npv)
    print('n fps: ', fp)
    print('n fns: ', fn)
    return sens, spec, acc, ppv, npv

def bootstrap_metric(df, thresh, n_resamples, metric):
    assert metric in ['sens', 'spec', 'ppv', 'npv', 'acc', 'pAUC']
    strapped_metric = []
    for _ in tqdm(range(n_resamples)):
        new_df = resample(df)
        pred = new_df['pt_ca_score'] > thresh
        y = new_df['cancer_ep']
        cm = confusion_matrix(y_true=y, y_pred=pred)
        tn, fp, fn, tp = cm.ravel()
        if metric == 'sens':
            met = tp / (tp + fn)
        elif metric == 'spec':
            met = tn / (tn + fp)
        elif metric == 'ppv':
            met = tp / (tp + fp)
        elif metric == 'npv':
            met = tn / (tn + fn)
        elif metric == 'acc':
            met = (tn + tp)/ cm.sum()
        elif metric == 'pAUC':
            met = roc_auc_score(
                new_df['cancer_ep'], new_df['pt_ca_score'],
                max_fpr=0.4)
            
        strapped_metric.append(met)
    ord_metrics = sorted(strapped_metric)
    lci = np.round(ord_metrics[int(0.025 * len(ord_metrics))], 3)
    uci = np.round(ord_metrics[int(0.975 * len(ord_metrics))], 3)
    return (lci, uci)

def plot_AUC(title, truths, probabilities, ci=False, n=False):
    fpr, tpr, roc_thresholds = roc_curve(truths, probabilities)

    pt_AUC = auc(fpr, tpr)
    if ci:
        AUCtext = 'AUC: '+str(np.round(pt_AUC, decimals=1)) + str(ci)
    else:
        AUCtext = 'AUC: '+str(np.round(pt_AUC, decimals=1))
    if n:
        AUCtext += ' \n n='+str(n)
        
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    plt.figure()
    plt.title(title)
    plt.plot(fpr, tpr)
    plt.text(0.42, 0.15, AUCtext, fontsize=14,
            verticalalignment='top', bbox=props)
    plt.xlabel('1-Specificity / False Positive Rate')
    plt.ylabel('Sensitivity / True Positive Rate')
    plt.show()
    

def results(df, CM=False, threshold=0.1875):
    fpr, tpr, roc_thresholds = roc_curve(df['cancer_ep'], df['pt_ca_score'])
    ci = bootstrap_AUC(df, channels=False, n_resamples=10000)
    ci = (round(ci[0]*100, 1), round(ci[1]*100, 1))
    AUC = round(auc(fpr, tpr) * 100, 1)
    pAUC = roc_auc_score(df['cancer_ep'], df['pt_ca_score'], max_fpr=0.4)
    if CM: #confusion matrix:
        cm = confusion_matrix(
            y_true=df['pt_ca_score'] >= threshold,
            y_pred=df['cancer_ep'])
        _metrics(true=df['cancer_ep'],
                 pred=df['pt_ca_score'] >= threshold)
        return {'fpr':fpr, 'tpr':tpr, 'AUC':AUC, 'CI':ci, 'pAUC':pAUC, 'CM':cm}
    else:
        return {'fpr':fpr, 'tpr':tpr, 'AUC':AUC, 'CI':ci, 'pAUC':pAUC}


    
def results_from_frame(PATH, VENDF_PATH=False, ca_v_norm=False, CM=False, return_df=False):
    """ loads csv, adds patient-wise prediction scores, 
    links to bssa data and gt, can exclude benign pts """
    df = pd.read_csv(PATH)
    df = add_result_columns(df, VENDF_PATH, ca_v_norm=ca_v_norm)
    if return_df:
        return results(df, CM=CM), df
    else:
        return results(df, CM=CM)



def sim_only_reader(df, ca_thresh):
    '''takes model output df, ca and benign thresholds and calculates 
    performance if model was sole reader
    '''
    print('There were {} pts recalled and biopsied for benign tumours'.format(sum(df['benign_tumour_ep'])))
    
    ca_test_pos = [int(x) for x in df['pt_ca_score'] > ca_thresh]
    ca_pos = df['cancer_ep'].tolist()
    tn, fp, fn, tp = confusion_matrix(ca_pos, ca_test_pos).ravel()
    sens = (tp / (tp+fn))
    spec = (tn / (tn + fp))
    ppv = tp / (tp + fp)
    print('\n if ca_threshold is:\n', ca_thresh)
    print('Sensitivity:', '%.4f' % sens)
    print('Specificity:', '%.4f' % spec)
    print('ppv:', '%.4f' % ppv)
    print('and there are n={} false positives'.format(fp))
    
def test_thresholds(df, thresholds, channel):
    thresh_df = pd.DataFrame(index=thresholds, columns=['sensitivity', 'specificity', 'PPV', 'f1', 'f2', 'f3', 'accuracy'])
    thresh_df.index.name = 'threshold'
    for i, thresh in enumerate(thresholds):
        thresh_col = 'ca_bin_pred_' + channel + 'thresh_'+str(thresh)
        thresh_df.iloc[i,:] = _metrics(
                true=df['cancer_ep'], pred=df[thresh_col]
                )
    return thresh_df