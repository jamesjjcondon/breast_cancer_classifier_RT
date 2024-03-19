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
import pandas as pd
from src.utilities.all_utils import load_df, decode_col
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
from src.constants import REPODIR, CSVDIR, BASECOLS2, DATADIR, NVMEDIR #, RESDIR
from src.eval.helpers import _metrics, test_thresholds, plot_AUC, bootstrap_AUC, add_result_columns
import seaborn as sns
sns.set()

from IPython import embed
#%%
# Directory where prediction csvs have been saved (with src/modeling/run_model.py --outpath=(dir)):

RESDIR = os.path.join(DATADIR, 
                      #'train_models/ckpts_21Mar_wd2e-06_v1/preds_5050_21Mar_wd2e-06v1'
                      'test_ims_master/preds/small_matched_3c_NYU_init_preds.csv' #AIML
                      )

# RESDIR2 = os.path.join(DATADIR, 
#                       #'train_models/ckpts_21Mar_wd2e-06_v1/preds_5050_21Mar_wd2e-06v1'
#                       'test_ims_master/NY_16-8pc_incidence/NYU_model_sf2' #AIML
#                       )

# filepath of df holding each images vendor/manufacturer (with src/utilities/save_vendors.py):
VENDF_PATH = os.path.join(DATADIR, 'TTS_logs/dfs_with_vendors/balanced/test_df.csv')

#%%

class eval_stratum:
    def __init__(self, fp : str, saving : bool):
        # load inference output .csv:
        self.df = add_result_columns(pd.read_csv(fp), VENDF_PATH=False)
        self.norms = self.df[(self.df['HIST_OUTCOME.string()'].isin([0., 3., 4.])) & (self.df['AX_MAMM_DOM_CAT_LESA_EV1.string()'] == 4)]
        self.saving = saving
        self.fp = fp
    
    def results(self, df):
        fpr, tpr, roc_thresholds = roc_curve(df['cancer_ep'], df['pt_ca_score'])
        ci = bootstrap_AUC(df, channels=False, n_resamples=10000)
        ci = (round(ci[0]*100, 1), round(ci[1]*100, 1))
        AUC = round(auc(fpr, tpr) * 100, 1)
        return {'fpr':fpr, 'tpr':tpr, 'AUC':AUC, 'CI':ci}
    
    def vis(self, res, title, strata, labels):
        assert isinstance(res, list)
        assert len(res) == len(labels)
        fig = plt.figure(figsize=(7,6))
        plt.title(title,
                  loc='left',
                  fontdict={'fontsize': 14})
        
        # embed()
        
        for line, label in zip(res, labels):
            t = '\nAUC: '+str(np.round(line['AUC'], decimals=1)) + " " + str(line['CI']) 
            if label is not None:
                plt.plot(line['fpr'], line['tpr'], linewidth=3, label=label + t)
            else:
                plt.plot(line['fpr'], line['tpr'], linewidth=3,
                         label="AUC: "+str(np.round(line['AUC'], decimals=1)) + " " + str(line['CI']) )
                
        plt.legend(loc='lower right', fontsize='medium')
        plt.xlabel('1-Specificity / False Positive Rate')
        plt.ylabel('Sensitivity / True Positive Rate')
        
        if self.saving:
            plt.savefig(
                    fname=os.path.join(*[REPODIR, 'figures_and_tables', strata+'.png']),
                    bbox_inches='tight')
        plt.show()
    
    def strat_to_df(self, strata: str, VENDF_PATH=False):
        """ Slice main df for strata patients and plot AUROC """
        if strata == 'IBCvDCIS':
            self.ibc = self.df[self.df['IBC'] == 1]
            self.dcis = self.df[self.df['DCIS'] == 1]
            self.ibc = self.results(pd.concat([self.ibc, self.norms]))
            self.dcis = self.results(pd.concat([self.dcis, self.norms]))
            self.vis([self.ibc, self.dcis],
                     title="Receiver Operating Characteristic Curves for Malignancy \n" + \
                     "Differentation: Invasive Breast Cancer (IBC) and \nDuctal Carcinomna In-situ (DCIS)", 
                     strata=strata,
                     labels=['IBC', 'DCIS']
                     ) 
            
        elif strata == 'age50':
            self.sub50 = self.df[self.df['AAS'] < 50]
            self.o50 = self.df[self.df['AAS'] >= 50]
            self.sub50res = self.results(self.sub50)
            self.o50res = self.results(self.o50)
            self.vis([self.sub50res, self.o50res], 
                     title="Receiver Operating Characteristic Curves for \n" + \
                     "Malignancy Differentation by Age", 
                     strata=strata,
                     labels=['Under 50', '50 or Older']
                     )            
            
        elif strata == 'calc':
            #    # create "normal" sample consisting of benign calcifications (for comparison with malignant calcification):
            nyhm15_retrained_name = os.path.join(
                    DATADIR, 'test_ims_master/preds/small_matched_16pc_incidence_3c_retrained_preds.csv')
            df = pd.read_csv(nyhm15_retrained_name)
            df = add_result_columns(df, VENDF_PATH=False)
            ben_calc = df[(df['HIST_OUTCOME.string()'] == 0.) & (df['AX_MAMM_DOM_CAT_LESA_EV1.string()'] == 1)]
            print('\n\tusing {} benign_calcification episodes'.format(ben_calc.shape[0]))
            calcs = self.df[self.df['AX_WU_DOM_CAT_LESA_EV1.string()'] == 2]
            print('\n\tusing {} malig calc episodes'.format(calcs.shape[0]))

            self.calc = pd.concat([calcs, ben_calc])
            self.res = self.results(self.calc)
            t1 = 'AUC: '+str(np.round(self.res['AUC'], decimals=3)) + '\n' + str(self.res['CI']) 
            fig = plt.figure(figsize=(7,6))
            plt.title(label="Receiver Operating Characteristic Curve for Calcification:\n" + \
                     "Malignant Versus Benign Calcification",
                     loc='left',
                     fontdict={'fontsize': 14})
            plt.plot(self.res['fpr'], self.res['tpr'], linewidth=3, label=t1)
            plt.legend(loc='lower right', fontsize='medium')
            plt.xlabel('1-Specificity / False Positive Rate')
            plt.ylabel('Sensitivity / True Positive Rate')
#            
            if self.saving:
                plt.savefig(fname=os.path.join(*[REPODIR, 'figures_and_tables', 'malig_v_benign_calc.png']),
                    bbox_inches='tight')
            plt.show()
#            import IPython; IPython.embed()
            
        elif strata == 'non-calc_findings':
            self.dm = self.df[(self.df['AX_WU_DOM_CAT_LESA_EV1.string()'] == 1) & (self.df['HIST_OUTCOME.string()'] == 1)] # discrete mass & malignant
            self.onsad = self.df[(self.df['AX_WU_DOM_CAT_LESA_EV1.string()'] == 3) & (self.df['HIST_OUTCOME.string()'] == 1)] # other, non-specific density and arch dist & malignant 
            self.stelm = self.df[(self.df['AX_WU_DOM_CAT_LESA_EV1.string()'] == 4) & (self.df['HIST_OUTCOME.string()'] == 1)] # stellates, multiple masses & malignant
            x = self.dm.shape[0]
            y = self.onsad.shape[0]
            z = self.stelm.shape[0]
            print('\n\t', x, y, z, 'discrete mass', 'other/non-specific-density/architectural distortion', 'stellates')
            self.disc = self.results(pd.concat([self.dm, self.norms]))
            self.ad = self.results(pd.concat([self.onsad, self.norms]))
            self.stel = self.results(pd.concat([self.stelm, self.norms]))
            self.vis([self.disc, self.ad, self.stel], 
                     title='Receiver Operating Characteristic Curves \nfor Malignancy differentation by Assessment Findings', 
                     strata=strata,
                     labels=['Discrete masses', "Non-specific Density, AD and 'other'", 'Stellates and Multiple Masses'])
            
        elif strata == 'vendor':
            self.df = add_result_columns(pd.read_csv(self.fp), VENDF_PATH=VENDF_PATH)
            self.philips = self.df[self.df['vendor'].str.contains('Philips')]
            self.sectra = self.df[self.df['vendor'].str.contains('Sectra')]
            self.pres = self.results(self.philips)
            self.sres = self.results(self.sectra)
            self.vis([self.pres, self.sres],
                     title="Receiver Operating Characterisitc Curves for \n" + 
                     "Malignancy Differentiation by Vendor (manufacturer)",
                     strata=strata,
                     labels=['Philips', 'Sectra'])

        elif strata == 'rad_size':
            # left ( is exclusive, right ] is inclusive
            size = 'AX_RADIO_SIZE_LESA_EV1.int32()'
            self.df_five_or_less = self.df[self.df[size] == 0]
            print(self.df_five_or_less.shape[0], '<5')
            
            self.df_more_than_5_less_eq_10 = self.df[self.df[size].isin([8,9,10,11,12])]
            print(self.df_more_than_5_less_eq_10.shape[0], '6 - <= 10')
            
            self.df_more_than_10_less_eq_14 = self.df[self.df[size].isin([1,2])]
            print(self.df_more_than_10_less_eq_14.shape[0], '11 - <= 14')
            
            self.df_more_than_14_less_eq_20 = self.df[self.df[size].isin([3,4])]
            print(self.df_more_than_14_less_eq_20.shape[0], '15 - <= 20')
            
            self.df_more_than_20 = self.df[self.df[size].isin([5,6,7])]
            print(self.df_more_than_20.shape[0], '>20')
            
           
            self.five_or_less = self.results(pd.concat([self.df_five_or_less, self.norms]))
            self.more_than_5_less_eq_10 = self.results(pd.concat([self.df_more_than_5_less_eq_10, self.norms]))
            self.more_than_10_less_eq_14 = self.results(pd.concat([self.df_more_than_10_less_eq_14, self.norms]))
            self.more_than_14_less_eq_20 = self.results(pd.concat([self.df_more_than_14_less_eq_20, self.norms]))
            self.more_than_20 = self.results(pd.concat([self.df_more_than_20, self.norms]))
            
            self.vis([self.five_or_less, 
                      self.more_than_5_less_eq_10, 
                      self.more_than_10_less_eq_14,
                      self.more_than_14_less_eq_20,
                      self.more_than_20
                      ],
                     title="Receiver Operating Characteristic Curves \nfor Malignancy" + \
                     " by Radiological Size (mm)", 
                     strata=strata,
                     labels=['<=5', '6 - <=10', '11 - <=14', '15 - <=20', '>20']
                     )
        
        elif strata == 'path_inv_size':
            self.inv_15_or_less = self.df[(self.df['HIST_OUTCOME.string()'] == 1) & (self.df['invasive_>15mm'] == 0)]
            self.inv_more_than_15 = self.df[(self.df['HIST_OUTCOME.string()'] == 1) & (self.df['invasive_>15mm'] == 1)]
            self.inv_15_or_less_res = self.results(pd.concat([self.inv_15_or_less, self.norms]))
            self.inv_more_than_15_res = self.results(pd.concat([self.inv_more_than_15, self.norms]))
            
            #embed()
            self.vis([self.inv_15_or_less_res, 
                      self.inv_more_than_15_res
                      ],
                     title="Receiver Operating Characteristic Curves \nfor Invasive Malignancy" + \
                     " by Pathological Size (mm)",
                     strata=strata,
                     labels=['<=15mm', '>15mm']
                     )
    
#%%    
i = eval_stratum(fp=RESDIR, saving=True)

i.strat_to_df('age50')
i.strat_to_df('IBCvDCIS')
i.strat_to_df('calc')
i.strat_to_df('non-calc_findings')
i.strat_to_df('rad_size')
i.strat_to_df('path_inv_size')
i.strat_to_df('vendor', VENDF_PATH='/data/james/NYU_retrain/TTS_logs/dfs_with_vendors/balanced/test_df.csv')
