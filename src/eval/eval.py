#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:17:13 2019

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
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from src.constants import REPODIR, DATADIR
from src.eval.helpers import bootstrap_metric, _metrics, results_from_frame
from src.utilities.all_utils import sens_spec_matrix_df

from IPython import embed
#%%
DATADIR = DATADIR +'/preds'
# Directory were prediction csvs have been saved (with src/modeling/run_model.py --outpath=(dir)):

# balanced
# NYU1
# naive:
NYU1_naive_fp = os.path.join(
        DATADIR, 'small_matched_balanced_1c_NYUC_naive_preds.csv')
# from scratch:
NYU1_from_scratch_fp = os.path.join(
        DATADIR, 'small_matched_1c_from_scratch_preds.csv')
# tl:
NYU1_tl_fp = os.path.join(
        DATADIR, 'small_matched_1c_NYU_init_preds.csv')

# NYU2:
# naive:
NYU2_naive_fp = os.path.join(
        DATADIR, 'small_matched_3c_NYU_naive_preds.csv')

# naive, small_matched:
NYU2_naive_sm_fp = os.path.join(
        DATADIR, 'small_matched_balanced_3c_NYU_naive_preds.csv')

# from scratch:
NYU2_from_scratch_fp = os.path.join(
        DATADIR, 'small_matched_3c_from_scratch_preds.csv')

# tl:
NYU2_tl_fp = os.path.join(
        DATADIR, 'small_matched_3c_NYU_init_preds.csv')

# imbalanced:
NYU2_tl_13pc_fp = os.path.join(
        DATADIR, 'small_matched_16pc_incidence_3c_retrained_preds.csv')

# filepath of df holding each images vendor/manufacturer (with src/utilities/save_vendors.py):
VENDF_PATH = None #'/data/james/NYU_retrain/TTS_logs/dfs_with_vendors/balanced/test_df.csv'
           
#%%
        
def model_bin_pred_col(df, ca_or_benign):
    """ adds a final binary prediction based on highest class probability for each view
    If any one view's highest probability is for malignancy/benign tumour, score for episode/patient is 1"""
    if ca_or_benign == 'ca':
        search = 'malignant'
    else:
        search = 'benign'
    cols = [x for x in df.columns if search in x and 'MCPs' in x]
    col_name = 'model_max_class_' + search
    df[col_name] = np.max(df[cols], axis=1)
    return df
        
def collect_results():  
    """ loads inference csvs and links to bssa data and ground truth
    #    ca_v_norm=True removes benign patients (hack for better AUC)
    #    returns dictionary of result dicts """
    # store results. without and with heatmaps, naive, 
    # from scratch / random_init and with transfer_learning (tl) / NYU_init:
    NYU1_naive = results_from_frame(
            NYU1_naive_fp, VENDF_PATH, ca_v_norm=False, CM=True)
    NYU1_scratch = results_from_frame(
            NYU1_from_scratch_fp, VENDF_PATH, ca_v_norm=False, CM=True)
    NYU1_tl = results_from_frame(
            NYU1_tl_fp, VENDF_PATH, ca_v_norm=False, CM=True)
    NYU2_naive = results_from_frame(
            NYU2_naive_sm_fp, VENDF_PATH, ca_v_norm=False, CM=True)
    NYU2_scratch = results_from_frame(
            NYU2_from_scratch_fp, VENDF_PATH, ca_v_norm=False, CM=True)
    NYU2_tl, lts1_df = results_from_frame(
            NYU2_tl_fp, VENDF_PATH, ca_v_norm=False, CM=True, return_df=True)   
    # NYU2_tl_13pc, lts2_df = results_from_frame(
    #         NYU2_tl_13pc_fp, VENDF_PATH=None, ca_v_norm=False, CM=True, return_df=True)
    return {
            'NYU1_naive':NYU1_naive,
            'NYU1_scratch':NYU1_scratch,
            'NYU1_tl':NYU1_tl,
            'NYU2_naive':NYU2_naive,
            'NYU2_scratch':NYU2_scratch,
            'NYU2_tl': NYU2_tl,
            # 'NYU2_tl_13pc':NYU2_tl_13pc
            }, lts1_df

#%%   
res, LTS1_df = collect_results()    
embed()

#%%
""" NYU2"""

plt.figure(figsize=(7,5))
plt.title(
        label="Effect of NYU2 Retraining on ROC and AUC \n" + \
        "Differentiating Malignancy from Benign Lesions \nand Age-Matched Controls " + \
        "(Balanced Dataset)",
        loc='left',
        fontdict={'fontsize': 14})#,
        #pad=20)

# Off the shelf

auc_text = "Static NYU2 model\nAUC: "+ str(
    round(
        res['NYU2_naive']['AUC']/100, 2)
    )
    
ci = ' (95%CI ' + str(round(res['NYU2_naive']['CI'][0]/100,2)) + '-' + str(round(res['NYU2_naive']['CI'][1]/100,2)) + ')'

plt.plot(res['NYU2_naive']['fpr'],res['NYU2_naive']['tpr'], 
         linewidth=3,
         label=auc_text + ci)

# Transfer learning / re-trained

auc_text = "Retrained NYU2 model\nAUC: "+ str(
    round(
        res['NYU2_tl']['AUC']/100, 2)
    )
    
ci = ' (95%CI ' + str(round(res['NYU2_tl']['CI'][0]/100,2)) + '-' + str(round(res['NYU2_tl']['CI'][1]/100,2)) + ')'

plt.plot(res['NYU2_tl']['fpr'],res['NYU2_tl']['tpr'], 
         linewidth=3,
         label=auc_text + ci)

plt.xlabel('1-Specificity / False Positive Rate')
plt.ylabel('Sensitivity / True Positive Rate')
plt.legend(loc='lower right', fontsize='medium')
plt.savefig(os.path.join(
        REPODIR, 'figures_and_tables/NYU2_effect_of_retraining.png'),
        bbox_inches='tight')

plt.show()

#%%
""" NYU1 """
plt.figure(figsize=(7,5))
plt.title(
        label="Effect of NYU1 Retraining on ROC and AUC \n" + \
        "Differentiating Malignancy from Benign Lesions \nand Age-Matched Controls " + \
        "(Balanced Dataset)",
        loc='left',
        fontdict={'fontsize': 14})#,
        #pad=20)

auc_text = "Static NYU1 model\nAUC: "+ str(
    round(
        res['NYU1_naive']['AUC']/100, 2)
    )

ci = ' (95%CI ' + str(round(res['NYU1_naive']['CI'][0]/100,2)) + '-' + str(round(res['NYU1_naive']['CI'][1]/100,2)) + ')'

plt.plot(res['NYU1_naive']['fpr'],res['NYU1_naive']['tpr'], 
         linewidth=3,
         label=auc_text + ci)

auc_text = 'Retrained NYU1 model\nAUC: '+ str(
    round(
        res['NYU1_tl']['AUC']/100, 2)
    )

ci = ' (95%CI ' + str(round(res['NYU1_tl']['CI'][0]/100,2)) + '-' + str(round(res['NYU1_tl']['CI'][1]/100,2)) + ')'

plt.plot(res['NYU1_tl']['fpr'],res['NYU1_tl']['tpr'], 
         linewidth=3,
         label=auc_text + ci)

plt.xlabel('1-Specificity / False Positive Rate')
plt.ylabel('Sensitivity / True Positive Rate')
plt.legend(loc='lower right', fontsize='medium')
plt.savefig(os.path.join(
        REPODIR, 'figures_and_tables/NYU1_effect_of_retraining.png'),
        bbox_inches='tight')

plt.show()
#%%
""" Additional Test - NYU2 transfer learning by incidence """
plt.figure(figsize=(7,5))
plt.title(
        label="Receiver Operating Characteristic Curves \n" + \
        "Differentiating Malignancy from Benign Lesions \n" + \
        "and Age-Matched Controls by Malignancy Prevalence",
        loc='left',
        fontdict={'fontsize': 14})

plt.plot(res['NYU2_tl']['fpr'],res['NYU2_tl']['tpr'], 
         linewidth=3,
         label='Retrained, \nBalanced  44% Malignancy Clients. \nAUC: '+ str(res['NYU2_tl']['AUC']) + ' ' + str(res['NYU2_tl']['CI']))

plt.plot(res['NYU2_tl_13pc']['fpr'],res['NYU2_tl_13pc']['tpr'], 
         linewidth=3,
         label='Retrained, \n13% Malignancy Clients, \nAUC: '+ str(res['NYU2_tl_13pc']['AUC']) + ' ' + str(res['NYU2_tl_13pc']['CI']))

plt.xlabel('1-Specificity / False Positive Rate')
plt.ylabel('Sensitivity / True Positive Rate')
plt.legend(loc='lower right', fontsize='medium')
plt.savefig(os.path.join(REPODIR, 'figures_and_tables/AUROC_Retrained_by_incidence.png'),
          bbox_inches='tight')
plt.show()
#plt.plot(out['nyhm_sf2_mid']['fpr'],out['nyhm_sf2_mid']['tpr'], 
#         linewidth=3,
#         label='Half-size with heatmaps, \ncentre image crop. AUC: '+ '%.3f' % out['nyhm_sf2_mid']['AUC'] + str(out['nyhm_sf2_mid']['CI']))

#%%
# what if NYU didn't exist:
plt.figure(figsize=(7,5))
plt.title(
        label="Cumulative Effect of Shared NYU Weights and Pixel-level\n" + \
        "Heatmaps on Receiver Operating Characteristic Curves \n" + \
        "Differentiating Malignancy from Benign Lesions \n" + \
        "and Age-Matched Controls (Balanced Test Set)",
        loc='left',
        fontdict={'fontsize': 14})

plt.plot(res['NYU1_scratch']['fpr'],res['NYU1_scratch']['tpr'], 
         linewidth=3,
         label='Images Only, \nWeights Randomly Initialised\nAUC: '+ str(res['NYU1_scratch']['AUC']) + ' ' +str(res['NYU1_scratch']['CI']))

plt.plot(res['NYU2_tl']['fpr'],res['NYU2_tl']['tpr'], 
         linewidth=3,
         label='Images and Heatmaps, \nTransfer Learning. \nAUC: '+ str(res['NYU2_tl']['AUC']) + ' ' + str(res['NYU2_tl']['CI']))

plt.xlabel('1-Specificity / False Positive Rate')
plt.ylabel('Sensitivity / True Positive Rate')
plt.legend(loc='lower right', fontsize='medium')
plt.savefig(os.path.join(REPODIR, 'figures_and_tables/Cumulative_effect_of_NYU_system.png'),
            bbox_inches='tight')
plt.show()

#%%
# NYU 1
plt.figure(figsize=(7,5))
plt.title('Effect of NYU1 Retraining on AUROC',
          loc='left',
          fontdict={'fontsize': 14})
plt.plot(res['NYU1_naive']['fpr'], res['NYU1_naive']['tpr'],
         linewidth=3,
         label='NYU1 static\nAUC: '+ str(res['NYU1_naive']['AUC']) + ' ' +str(res['NYU1_naive']['CI']))
plt.plot(res['NYU1_scratch']['fpr'], res['NYU1_scratch']['tpr'],
         linewidth=3,
         label='NYU1 from scratch\nAUC: '+ str(res['NYU1_scratch']['AUC']) + ' ' +str(res['NYU1_scratch']['CI']))
plt.plot(res['NYU1_tl']['fpr'], res['NYU1_tl']['tpr'],
         linewidth=3,
         label='NYU1 with transfer learning\nAUC: '+ str(res['NYU1_tl']['AUC']) + ' ' +str(res['NYU1_tl']['CI']))
        
plt.xlabel('1-Specificity / False Positive Rate')
plt.ylabel('Sensitivity / True Positive Rate')
plt.legend(loc='lower right', fontsize='medium')
plt.savefig(os.path.join(REPODIR, 'figures_and_tables/Effect_of_retraining_NYU1.png'),
            bbox_inches='tight')
plt.show()
#%%
# NYU2
plt.figure(figsize=(7,5))
plt.title('Effect of NYU2 Retraining on AUROC',
          loc='left',
          fontdict={'fontsize': 14})
plt.plot(res['NYU2_naive']['fpr'], res['NYU2_naive']['tpr'],
         linewidth=3,
         label='NYU2 static\nAUC: '+ str(res['NYU2_naive']['AUC']) + ' ' +str(res['NYU2_naive']['CI']))
plt.plot(res['NYU2_scratch']['fpr'], res['NYU2_scratch']['tpr'],
          linewidth=3,
          label='NYU2 from scratch\nAUC: '+ str(res['NYU2_scratch']['AUC']) + ' ' +str(res['NYU2_scratch']['CI']))
plt.plot(res['NYU2_tl']['fpr'], res['NYU2_tl']['tpr'],
         linewidth=3,
         label='NYU2 with transfer learning\nAUC: '+ str(res['NYU2_tl']['AUC']) + ' ' +str(res['NYU2_tl']['CI']))
plt.xlabel('1-Specificity / False Positive Rate')
plt.ylabel('Sensitivity / True Positive Rate')
plt.legend(loc='lower right', fontsize='medium')
plt.savefig(os.path.join(REPODIR, 'figures_and_tables/Effect_of_retraining_NYU2.png'),
            bbox_inches='tight')
plt.show()

#%%
# Naive models
plt.figure(figsize=(7,5))
plt.title('Effect of Heatmaps on Static Models')
plt.plot(res['NYU1_naive']['fpr'], res['NYU1_naive']['tpr'],
         linewidth=3,
         label='NYU1 static\nAUC: '+ str(res['NYU1_naive']['AUC']) + ' ' +str(res['NYU1_naive']['CI']))

plt.plot(res['NYU2_naive']['fpr'], res['NYU2_naive']['tpr'],
         linewidth=3,
         label='NYU2 static\nAUC: '+ str(res['NYU2_naive']['AUC']) + ' ' +str(res['NYU2_naive']['CI']))
plt.xlabel('1-Specificity / False Positive Rate')
plt.ylabel('Sensitivity / True Positive Rate')
plt.legend(loc='lower right', fontsize='medium')
plt.savefig(os.path.join(REPODIR, 'figures_and_tables/Effect_of_Heatmaps_Static.png'),
            bbox_inches='tight')
plt.show()

#%%

# scratch
plt.figure(figsize=(7,5))
plt.title('Effect of Heatmaps on Training from Scratch')
plt.plot(res['NYU1_scratch']['fpr'], res['NYU1_scratch']['tpr'],
         linewidth=3,
         label='NYU1 from scratch\nAUC: '+ str(res['NYU1_scratch']['AUC']) + ' ' +str(res['NYU1_scratch']['CI']))
plt.plot(res['NYU2_scratch']['fpr'], res['NYU2_scratch']['tpr'],
         linewidth=3,
         label='NYU2 from scratch\nAUC: '+ str(res['NYU2_scratch']['AUC']) + ' ' +str(res['NYU2_scratch']['CI']))
plt.xlabel('1-Specificity / False Positive Rate')
plt.ylabel('Sensitivity / True Positive Rate')
plt.legend(loc='lower right', fontsize='medium')
plt.savefig(os.path.join(REPODIR, 'figures_and_tables/Effect_of_Heatmaps_Scratch.png'),
            bbox_inches='tight')
plt.show()
#%%

# tl:
plt.figure(figsize=(7,5))
plt.title('Effect of Heatmaps on Retraining with Transfer Learning')
plt.plot(res['NYU1_tl']['fpr'], res['NYU1_tl']['tpr'],
         linewidth=3,
         label='NYU1 with transfer learning\nAUC: '+ str(res['NYU1_tl']['AUC']) + ' ' +str(res['NYU1_tl']['CI']))
plt.plot(res['NYU2_tl']['fpr'], res['NYU2_tl']['tpr'],
         linewidth=3,
         label='NYU2 with transfer learning\nAUC: '+ str(res['NYU2_tl']['AUC']) + ' ' +str(res['NYU2_tl']['CI']))
plt.xlabel('1-Specificity / False Positive Rate')
plt.ylabel('Sensitivity / True Positive Rate')
plt.legend(loc='lower right', fontsize='medium')
plt.savefig(os.path.join(REPODIR, 'figures_and_tables/Effect_of_Heatmaps_tl.png'),
          bbox_inches='tight')
plt.show()

#%%
one_naive = sens_spec_matrix_df('NYU1_naive', res, .9)
one_scratch = sens_spec_matrix_df('NYU1_scratch', res, .9)
one_tl = sens_spec_matrix_df('NYU1_tl', res, .9)
two_naive = sens_spec_matrix_df('NYU2_naive', res, .9)
two_scratch = sens_spec_matrix_df('NYU2_scratch', res, .9)
two_tl = sens_spec_matrix_df('NYU2_tl', res, .9)




_metrics(LTS1_df['cancer_ep'], LTS1_df['pt_ca_score'] > 0.175)


#%%
NYU1_naive, n1naive = results_from_frame(
            NYU1_naive_fp, VENDF_PATH, ca_v_norm=False,
            CM=True, return_df=True)

_metrics(n1naive['cancer_ep'], n1naive['pt_ca_score'] > 0.175)

    #%%
NYU1_scratch, n1s = results_from_frame(
        NYU1_from_scratch_fp, VENDF_PATH, ca_v_norm=False, 
        CM=True, return_df=True)
_metrics(n1s['cancer_ep'], n1s['pt_ca_score'] > 0.175)

#%%

NYU1_tl, n1tl = results_from_frame(
        NYU1_tl_fp, VENDF_PATH, ca_v_norm=False, 
        CM=True, return_df=True)

_metrics(n1tl['cancer_ep'], n1tl['pt_ca_score'] > 0.175)

#%%

NYU2_naive, n2naive = results_from_frame(
        NYU2_naive_sm_fp, VENDF_PATH, ca_v_norm=False, 
        CM=True, return_df=True)

_metrics(n2naive['cancer_ep'], n2naive['pt_ca_score'] > 0.175)

#%%

NYU2_scratch, n2s = results_from_frame(
        NYU2_from_scratch_fp, VENDF_PATH, ca_v_norm=False,
        CM=True, return_df=True)

_metrics(n2s['cancer_ep'], n2s['pt_ca_score'] > 0.1875)

    #%%
NYU2_tl, n2tl = results_from_frame(
        NYU2_tl_fp, VENDF_PATH, ca_v_norm=False, 
        CM=True, return_df=True)  

sens, spec, acc, ppv, npv = _metrics(n2tl['cancer_ep'], n2tl['pt_ca_score'] > 0.1875) 

#%%
lci, uci = bootstrap_metric(n2tl, thresh = 0.1875, n_resamples=10000, metric='sens')

# NYU2_tl_13pc, lts2_df = results_from_frame(
#         NYU2_tl_13pc_fp, VENDF_PATH=None, ca_v_norm=False, CM=True, return_df=True)
      



