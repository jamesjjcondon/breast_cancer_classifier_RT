# Copyright (C) 2019 Nan Wu, Jason Phang, Jungkyu Park, Yiqiu Shen, Zhe Huang, Masha Zorin, 
#   Stanisław Jastrzębski, Thibault Févry, Joe Katsnelson, Eric Kim, Stacey Wolfson, Ujas Parikh, 
#   Sushma Gaddam, Leng Leng Young Lin, Kara Ho, Joshua D. Weinstein, Beatriu Reig, Yiming Gao, 
#   Hildegard Toth, Kristine Pysarenko, Alana Lewin, Jiyon Lee, Krystal Airola, Eralda Mema, 
#   Stephanie Chung, Esther Hwang, Naziya Samreen, S. Gene Kim, Laura Heacock, Linda Moy, 
#   Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of src.
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
"""
Defines constants used in src.
"""
import platform
machine = platform.node()

if machine == 'lambda-quad': #AIML
    CSVDIR = '/home/mlim-user/james/tmp' 
    TTSDIR = '/data/james/NYU_retrain/TTS_logs'
    REPODIR = '/home/mlim-user/james/nyukat2.0'
    RESDIR = '/data/james/NYU_retrain/test_ims/5050_preds_sf0'
    IMDIR = '/home/mlim-user/Documents/james/NYU_retrain/test_ims/renamed_dicoms' #images'
    NVMEDIR = '/nvme/james'
    DATADIR = '/data/james/NYU_retrain'
    VALDIR = '/nvme/james/val_ims_master'
    TESTDIR = '/nvme/james/test_ims_master'
    
elif machine == 'samlim-db': # intersect VM
    CSVDIR = '/data/samlim-db/bssa/drjc_dev/temp'

elif machine == 'DL136541': #AHMS SPH
    CSVDIR = '/data/james/bssa/temp'
#    TTSDIR = '/Data/james/NYU_val/Feb_TTS_logs'
    TTSDIR = '/data/james/NYU_Retrain/TTS_logs'
    REPODIR = '/home/james/mydev/nyukat2.0'
    #RESDIR = '/media/james/drjc_ext_HD/data/AHMS_NYURT_test/test_ims/5050_preds_sf2'
    NVMEDIR = DATADIR = VALDIR = '/data/james/NYU_Retrain'
#    IMDIR = '/Data/james/NYU_val/images_5050'
    #XHD = '/media/james/data/NYU_retrain'

elif machine == 'yoga': # laptop
    CSVDIR = '/data/bssa'
    DATADIR = NVMEDIR = VALDIR = '/data/NYU_retrain'
    TTSDIR = '/home/drjc/data/NYU_retrain/TTS_logs'
    REPODIR = '/home/drjc/dev/nyukat2.0'
    RESDIR = '/media/drjc/drjc_ext_HD/data/NYU_retrain/test_on_val/5050_preds'

elif machine == 'home':
    CSVDIR = '/data/bssa'
    DATADIR = NVMEDIR = VALDIR = '/data/bssa'
    REPODIR = '/home/drjc/mydev/nyukat2.0'
    RESDIR = '/data/bssa/preds'
    TTSDIR = None #'/data1/NYU_retrain/TTS_logs'
    
else:
    raise FileNotFoundError
    
class VIEWS:
    L_CC = "L-CC"
    R_CC = "R-CC"
    L_MLO = "L-MLO"
    R_MLO = "R-MLO"

    LIST = [L_CC, R_CC, L_MLO, R_MLO]

    @classmethod
    def is_cc(cls, view):
        return view in (cls.L_CC, cls.R_CC)

    @classmethod
    def is_mlo(cls, view):
        return view in (cls.L_MLO, cls.R_MLO)

    @classmethod
    def is_left(cls, view):
        return view in (cls.L_CC, cls.L_MLO)

    @classmethod
    def is_right(cls, view):
        return view in (cls.R_CC, cls.R_MLO)
    
#class BIG:


class VIEWANGLES:
    CC = "CC"
    MLO = "MLO"

    LIST = [CC, MLO]


class LABELS:
    LEFT_BENIGN = "left_benign"
    RIGHT_BENIGN = "right_benign"
    LEFT_MALIGNANT = "left_malignant"
    RIGHT_MALIGNANT = "right_malignant"

    LIST = [LEFT_BENIGN, RIGHT_BENIGN, LEFT_MALIGNANT, RIGHT_MALIGNANT]


class MODELMODES:
    VIEW_SPLIT = "view_split" # view-wise
    IMAGE = "image" # image-wise

# Native # h:w is 1.08952
native = {
    VIEWS.L_CC: (5355, 4915),
    VIEWS.R_CC: (5355, 4915),
    VIEWS.L_MLO: (5355, 4915),
    VIEWS.R_MLO: (5355, 4915),
}
# Large 3/4
three_forths = {
    VIEWS.L_CC: (4016, 3686),
    VIEWS.R_CC: (4016, 3686),
    VIEWS.L_MLO: (4016, 3686),
    VIEWS.R_MLO: (4016, 3686),
}

# Medium - approx halved and extra resize and padding to fit into NYU model
# h:w is 1.3784757
half = {
    VIEWS.L_CC: (2677, 1942),
    VIEWS.R_CC: (2677, 1942),
    VIEWS.L_MLO: (2974, 1748),
    VIEWS.R_MLO: (2974, 1748),
}

# Small
third = {
    VIEWS.L_CC: (1785, 1638),
    VIEWS.R_CC: (1785, 1638),
    VIEWS.L_MLO: (1785, 1638),
    VIEWS.R_MLO: (1785, 1638),
}
# Tiny
fifth = {
    VIEWS.L_CC: (1071, 983),
    VIEWS.R_CC: (1071, 983),
    VIEWS.L_MLO: (1071, 983),
    VIEWS.R_MLO: (1071, 983),
}

ca_codes = {0: 'ADENOCARCINOMA - NOS',
 8: 'CARCINOMA - ADENOID CYSTIC',
 9: 'CARCINOMA - INFILTRATING DUCT',
 10: 'CARCINOMA - INFILTRATING LOBULAR',
 11: 'CARCINOMA - MEDULLARY',
 12: 'CARCINOMA - MUCINOUS',
 13: 'CARCINOMA - NON-INFILTRATING - INTRADUCTAL - NOS',
 14: 'CARCINOMA - PAPILLARY - NOS (ADENOCARCINOMA)',
 15: 'CARCINOMA - TUBULAR - INVASIVE CRIBIFORM',
 33: 'INVASIVE MICROPAPILLARY CARCINOMA',
 37: 'METAPLASTIC CARCINOMA',
 38: 'METAPLASTIC CARCINOMA/PSEUDOSARCOMA',
 39: 'METASTATIC ADENOCARCINOMA',
 49: 'PLEOMORPHIC CARCINOMA IN SITU',
 51: 'SARCOMA',
 52: 'SOLID PAPILLARY CARCINOMA - IN SITU',
 53: 'SPINDLE CELL TUMOURS'}

benign_codes = {1: 'ADENOSIS - BLUNT DUCT',
 2: 'ADENOSIS - FLORID',
 3: 'ADENOSIS - NOS',
 4: 'ADENOSIS - SCLEROSING',
 5: 'BENIGN CALCIFICATION',
 6: 'BENIGN SOFT TISSUE TUMOUR',
 7: 'CALCIFICATION',
 17: 'COLUMNAR CELL ALTERATION',
 18: 'CYST - NOS',
 20: 'FAT NECROSIS',
 21: 'FIBROADENOMA - NOS',
 22: 'FIBROCYSTIC DISEASE - NOS',
 23: 'FIBROSIS OF THE BREAST',
 24: 'FOREIGN BODY GIANT CELL REACTION',
 25: 'GRANULAR CELL TUMOUR',
 27: 'HAMARTOMA - ADENO - FIBRO - LIPOMA',
 29: 'HYPERPLASIA - LOBULAR - ATYPICAL',
 30: 'HYPERPLASIA DUCTAL',
 32: 'INTRADUCT PAPILLOMATOSIS - ATYPICAL',
 34: 'LOBULAR CARCINOMA IN SITU - LCIS',
 40: 'MUCOCOELE-LIKE LESION',
 45: 'PAPILLOMA - INTRADUCTAL',
 46: 'PAPILLOMATOSIS - INTRADUCTAL - NOS - MULTIPLE',
 47: 'PHYLLODES - BENIGN - NOS'}

BASECOLS = [
        'ID.int32()',
        'PQ_DOB.string()',
         'AAS',
         'EPISODE_NUMBER.int32()',
         'SX_ACCESSION_NUMBER.int32()',
         'SX_DATE.string()',
         'SX_TOTAL_IMAGES_R.int32()',
         'SX_TOTAL_IMAGES_L.int32()',
         'SX_REAS_EXTRA.string()',
         'AX_ACCESSION_NUMBER1.int32()',
         'AX_ACCESSION_NUMBER2.int32()',
         'AX_ACCESSION_NUMBER3.int32()',
         'AX_WU_DOM_CAT_LESA_EV1.string()',
         'ASSESSMENT.string()',
         'Total_Eps',
         'Eps_N_to_last',
         'HIST_OUTCOME.string()',
         'HIST_SNOMED.string()',
         'Left_views_ca',
         'Left_views_benign',
         'PQ_IMPLANTS.string()',
         'PQ_SYMPTOMS.string()',
         'Right_views_ca',
         'Right_views_benign',
         'ca_pt',
         'cancer_ep',
         'AX_WU_DOM_CAT_LESA_EV1.string()',
         'TX_BREASTS_INVOLVED.string()'
                             ]


BASECOLS2 = [
        'ID.int32()',
        'PQ_DOB.string()',
        'new_names',
        'AN',
         'AAS',
#         'EPISODE_NUMBER.int32()',
#         'SX_ACCESSION_NUMBER.int32()',
#         'SX_DATE.string()',
#         'SX_TOTAL_IMAGES_R.int32()',
#         'SX_TOTAL_IMAGES_L.int32()',
#         'SX_REAS_EXTRA.string()',
#         'AX_ACCESSION_NUMBER1.int32()',
#         'AX_ACCESSION_NUMBER2.int32()',
#         'AX_ACCESSION_NUMBER3.int32()',
         'AX_WU_DOM_CAT_LESA_EV1.string()',
         'AX_WU_DOM_CAT_LESA_EV2.string()',
         'AX_WU_DOM_CAT_LESB_EV1.string()',
         'AX_WU_DOM_CAT_LESB_EV2.string()',
#         'ASSESSMENT.string()',
         'Total_Eps',
#         'Eps_N_to_last',
         'HIST_OUTCOME.string()',
         'HIST_SNOMED.string()',
         'HIST_STAGE.string()',
         'HIST_MALIG_GRADE_LESA.string()',
         'HIST_MALIG_GRADE_LESB.string()',
         'HIST_MALIG_INV_SIZE_LESA.int32()',
         'HIST_MALIG_INV_SIZE_LESB.int32()',
         'HIST_MALIG_NONINVAS_LESA.string()',
         'HIST_MALIG_NONINVAS_LESB.string()',
         'DCIS',
         'HIST_MALIG_DCIS_SIZE_LESA.int32()',
         'HIST_MALIG_DCIS_SIZE_LESB.int32()',
         'IBC',
         'Left_views_ca',
         'Left_views_benign',
         'Right_views_ca',
         'Right_views_benign',
         'ca_pt',
         'cancer_ep',
#         'TX_BREASTS_INVOLVED.string()'
                             ]
NONG_COLS = [
         'ID.int32()',
        'AAS',
 'AASI',
 'AAS_Over_50',
 'ASSESSMENT.string()',
 'AX_ACCESSION_NUMBER1.int32()',
 'AX_ACCESSION_NUMBER2.int32()',
 'AX_ACCESSION_NUMBER3.int32()',
 
 'AX_MAMM_DOM_CAT_LESA_EV1.string()',
 'AX_MAMM_DOM_CAT_LESA_EV2.string()',
 'AX_MAMM_DOM_CAT_LESA_EV3.string()',
 'AX_MAMM_DOM_CAT_LESB_EV1.string()',
 'AX_MAMM_DOM_CAT_LESB_EV2.string()',
 'AX_MAMM_LESA_EV1.string()',
 'AX_MAMM_LESA_EV2.string()',
 'AX_MAMM_LESA_EV3.string()',
 'AX_MAMM_LESB_EV1.string()',
 'AX_MAMM_LESB_EV2.string()',
 'AX_MAMM_RESULT_LESA_EV1.string()',
 'AX_MAMM_RESULT_LESA_EV2.string()',
 'AX_MAMM_RESULT_LESA_EV3.string()',
 'AX_MAMM_RESULT_LESB_EV1.string()',
 'AX_MAMM_RESULT_LESB_EV2.string()',
 
 'AX_RADIO_SIZE_LESA_EV1.int32()',
 'AX_RADIO_SIZE_LESA_EV2.int32()',
 'AX_RADIO_SIZE_LESA_EV3.int32()',
 'AX_RADIO_SIZE_LESB_EV1.int32()',
 'AX_RADIO_SIZE_LESB_EV2.int32()', 
 'AX_SUMM_LESA_EV1.string()',
 'AX_SUMM_LESA_EV2.string()',
 'AX_SUMM_LESA_EV3.string()',
 'AX_SUMM_LESB_EV1.string()',
 'AX_SUMM_LESB_EV2.string()',
 
 'AX_WU_DOM_CAT_LESA_EV1.string()',
 'AX_WU_DOM_CAT_LESA_EV2.string()',
 'AX_WU_DOM_CAT_LESA_EV3.string()',
 'AX_WU_DOM_CAT_LESB_EV1.string()',
 'AX_WU_DOM_CAT_LESB_EV2.string()',
 'AX_WU_LESA_EV1.string()',
 'AX_WU_LESA_EV2.string()',
 'AX_WU_LESA_EV3.string()',
 'AX_WU_LESB_EV1.string()',
 'AX_WU_LESB_EV2.string()',
 'AX_WU_OUTCOME_EV1.string()',
 'AX_WU_OUTCOME_EV2.string()',
 'AX_WU_OUTCOME_EV3.string()',
 
 'EPISODE_NUMBER.int32()',
 'Eps_N_to_last',
 'Eps_Post_SNOMED',
 'Eps_Pre_SNOMED',
 
 'HISTOPATHOLOGY.string()',
 'HIST_AXIL_DIS.string()',
 'HIST_AXIL_DIS_LESA.string()',
 'HIST_AXIL_DIS_LESB.string()',
 'HIST_AXIL_DIS_NODES_LESA.int32()',
 'HIST_AXIL_DIS_NODES_LESB.int32()',
 'HIST_AXIL_DIS_POS_NODES_LESA.int32()',
 'HIST_AXIL_DIS_POS_NODES_LESB.int32()',
 'HIST_DOM_LESION.string()',
 'HIST_FOCALITY.string()',
 'HIST_MALIG_DCIS_SIZE_LESA.int32()',
 'HIST_MALIG_DCIS_SIZE_LESB.int32()',
 'HIST_MALIG_DOM_CAT_LESA.string()',
 'HIST_MALIG_DOM_CAT_LESB.string()',
 'HIST_MALIG_GRADE_LESA.string()',
 'HIST_MALIG_GRADE_LESB.string()',
 'HIST_MALIG_INVAS_LESA.string()',
 'HIST_MALIG_INVAS_LESB.string()',
 'HIST_MALIG_INV_SIZE_LESA.int32()',
 'HIST_MALIG_INV_SIZE_LESB.int32()',
 'HIST_MALIG_LESA.string()',
 'HIST_MALIG_LESB.string()',
 'HIST_METHOD.string()',
 'HIST_NON_MALIG_DOM_CAT_LESA.string()',
 'HIST_NON_MALIG_DOM_CAT_LESB.string()',
 'HIST_OUTCOME.string()',
 'HIST_SNOMED.string()',
 'HIST_STAGE.string()',
 
 'IBC',
 'ID.int32()',
 'Left_views_ca',
 'MaxNUMREL1',
 'MaxNUMREL1_bin',
 'MaxNUMRELany',
 'MaxNUMRELany_bin',
 'Node_pos_>=3',
 
 'PQ_BREAST_PROBLEMS.string()',
 'PQ_DOB.string()',
 
 'PQ_FH_AGE_1.int32()',
 'PQ_FH_AGE_2.int32()',
 'PQ_FH_AGE_3.int32()',
 'PQ_FH_AGE_4.int32()',
 'PQ_FH_BREASTS_INVOLVED_1.string()',
 'PQ_FH_BREASTS_INVOLVED_2.string()',
 'PQ_FH_BREASTS_INVOLVED_3.string()',
 'PQ_FH_BREASTS_INVOLVED_4.string()',
 'PQ_FH_BREAST_CANCER.string()',
 'PQ_FH_RELATIONSHIP_1.string()',
 'PQ_FH_RELATIONSHIP_2.string()',
 'PQ_FH_RELATIONSHIP_3.string()',
 'PQ_FH_RELATIONSHIP_4.string()',
 'PQ_HRT.string()',
 'PQ_IMPLANTS.string()',
 'PQ_PREV_CANCER.string()',
 'PQ_PREV_CANCER_AGE.string()',
 'PQ_PREV_CANCER_LAT.string()',
 'PQ_SYMPTOMS.string()',
 'PRIMARY_TREATMENT.string()',
 'Pos_nodes_LESA+B',
 'R1_GRADE_LESA.string()',
 'R1_GRADE_LESB.string()',
 'R2_GRADE_LESA.string()',
 'R2_GRADE_LESB.string()',
 'R3_GRADE_LESA.string()',
 'R3_GRADE_LESB.string()',
 'Right_views_ca',
 'SURGICAL_TREATMENT.string()',
 'SX_ACCESSION_NUMBER.int32()',
 'SX_DATE.string()',
 'TX_BREASTS_INVOLVED.string()',
 'TX_LESA.string()',
 'TX_LESB.string()',
 'TX_RADIO.string()',
 'Total_Eps'
 ]

READER_COLS = [
 'R1_GRADE_LESA.string()',
 'R1_SIDE_LESA.string()',
 'R1_CALC_LESA.string()',
 'R1_STEL_LESA.string()',
 'R1_DISC_LESA.string()',
 'R1_MULT_LESA.string()',
 'R1_ARCH_LESA.string()',
 'R1_ASYM_LESA.string()',
 'R1_LYMP_LESA.string()',
 'R1_OTHER_LESA.string()',
 'R1_GRADE_LESB.string()',
 'R1_SIDE_LESB.string()',
 'R1_CALC_LESB.string()',
 'R1_STEL_LESB.string()',
 'R1_DISC_LESB.string()',
 'R1_MULT_LESB.string()',
 'R1_ARCH_LESB.string()',
 'R1_ASYM_LESB.string()',
 'R1_LYMP_LESB.string()',
 'R1_OTHER_LESB.string()',
 'R2_GRADE_LESA.string()',
 'R2_SIDE_LESA.string()',
 'R2_CALC_LESA.string()',
 'R2_STEL_LESA.string()',
 'R2_DISC_LESA.string()',
 'R2_MULT_LESA.string()',
 'R2_ARCH_LESA.string()',
 'R2_ASYM_LESA.string()',
 'R2_LYMP_LESA.string()',
 'R2_OTHER_LESA.string()',
 'R2_GRADE_LESB.string()',
 'R2_SIDE_LESB.string()',
 'R2_CALC_LESB.string()',
 'R2_STEL_LESB.string()',
 'R2_DISC_LESB.string()',
 'R2_MULT_LESB.string()',
 'R2_ARCH_LESB.string()',
 'R2_ASYM_LESB.string()',
 'R2_LYMP_LESB.string()',
 'R2_OTHER_LESB.string()',
 'R3_GRADE_LESA.string()',
 'R3_SIDE_LESA.string()',
 'R3_CALC_LESA.string()',
 'R3_STEL_LESA.string()',
 'R3_DISC_LESA.string()',
 'R3_MULT_LESA.string()',
 'R3_ARCH_LESA.string()',
 'R3_ASYM_LESA.string()',
 'R3_LYMP_LESA.string()',
 'R3_OTHER_LESA.string()',
 'R3_GRADE_LESB.string()',
 'R3_SIDE_LESB.string()',
 'R3_CALC_LESB.string()',
 'R3_STEL_LESB.string()',
 'R3_DISC_LESB.string()',
 'R3_MULT_LESB.string()',
 'R3_ARCH_LESB.string()',
 'R3_ASYM_LESB.string()',
 'R3_LYMP_LESB.string()',
 'R3_OTHER_LESB.string()'
 ]

All_snomeds = [
 'CARCINOMA - INFILTRATING DUCT',
 'CARCINOMA - NON-INFILTRATING - INTRADUCTAL - NOS',
 'CARCINOMA - INFILTRATING LOBULAR',
 'CARCINOMA - TUBULAR - INVASIVE CRIBIFORM',
 'HYPERPLASIA - INTRADUCTAL - ATYPICAL',
 'RADIAL SCAR',
 'PAPILLOMA - INTRADUCTAL',
 'LOBULAR CARCINOMA IN SITU - LCIS',
 'CARCINOMA - MUCINOUS',
 'FIBROCYSTIC DISEASE - NOS',
 'HYPERPLASIA - LOBULAR - ATYPICAL',
 'CARCINOMA - PAPILLARY - NOS (ADENOCARCINOMA)',
 'FIBROADENOMA - NOS',
 'HYPERPLASIA DUCTAL',
 'COLUMNAR CELL ALTERATION',
 'COLUMNAR CELL - ATYPIA - FLAT ATYPIA',
 'NORMAL TISSUE - NOS',
 'PHYLLODES - BENIGN - NOS',
 'CALCIFICATION',
 'PAPILLOMATOSIS - INTRADUCTAL - NOS - MULTIPLE',
 'ADENOSIS - SCLEROSING',
 'CARCINOMA - MEDULLARY',
 'BENIGN CALCIFICATION',
 'INTRADUCT PAPILLOMATOSIS - ATYPICAL',
 'CYST - NOS',
 'METAPLASTIC CARCINOMA/PSEUDOSARCOMA',
 'METAPLASTIC CARCINOMA',
 'MUCOCOELE-LIKE LESION',
 'METASTATIC ADENOCARCINOMA',
 'ADENOCARCINOMA - NOS',
 'FIBROSIS OF THE BREAST',
 'UNKNOWN',
 'FAT NECROSIS',
 'ADENOSIS - BLUNT DUCT',
 'PHYLLODES - BORDERLINE - NOS',
 'FOREIGN BODY GIANT CELL REACTION',
 'NON-SPECIFIC REACTIVE PROCESS',
 'BENIGN SOFT TISSUE TUMOUR',
 'ADENOSIS - NOS',
 'PLEOMORPHIC CARCINOMA IN SITU',
 'INVASIVE MICROPAPILLARY CARCINOMA',
 'ECTASIA - MAMMARY DUCT - PLASMA CELL MASTITIS',
 'METAPLASIA - APOCRINE',
 'GRANULAR CELL TUMOUR',
 'SARCOMA',
 'SPINDLE CELL TUMOURS',
 'ADENOSIS - FLORID',
 'INFLAMMATION - CHRONIC - NOS',
 "NON-HODGKIN'S LYMPHOMA",
 'CARCINOMA - ADENOID CYSTIC',
 'SOLID PAPILLARY CARCINOMA - IN SITU',
 'HAMARTOMA - ADENO - FIBRO - LIPOMA',
 'GRANULOMATOUS INFLAMMATION',
 'MASTITIS'
 ]
"""
NYU Dataset report re Ca vs bening:
     
    Malignant terms: 
        ductal carcinoma (2046), 
        ductal carcinoma in situ (1464), 
        invasive ductal carcinoma (1149),
        invasive carcinoma (557), 
        metastases (414), metastatic (224),
        invasive lobular carcinoma’, 180),
        adenocarcinoma (160),
        invasive mammary carcinoma (128),
        metastatic carcinoma, (117).
        """
cancers = [
 'CARCINOMA - INFILTRATING DUCT',
 'CARCINOMA - NON-INFILTRATING - INTRADUCTAL - NOS',
 'CARCINOMA - INFILTRATING LOBULAR',
 'CARCINOMA - TUBULAR - INVASIVE CRIBIFORM',
 'CARCINOMA - MUCINOUS',
 'CARCINOMA - PAPILLARY - NOS (ADENOCARCINOMA)',
 'CARCINOMA - MEDULLARY',
 'METAPLASTIC CARCINOMA/PSEUDOSARCOMA',
 'METAPLASTIC CARCINOMA',
 'METASTATIC ADENOCARCINOMA',
 'ADENOCARCINOMA - NOS',
 'PLEOMORPHIC CARCINOMA IN SITU',
 'INVASIVE MICROPAPILLARY CARCINOMA',
 'SARCOMA',
 'SPINDLE CELL TUMOURS',
 'CARCINOMA - ADENOID CYSTIC',
 'SOLID PAPILLARY CARCINOMA - IN SITU'
 ]

IBCs = [
 'ADENOCARCINOMA - NOS',
 'CARCINOMA - ADENOID CYSTIC',
 'CARCINOMA - INFILTRATING DUCT',
 'CARCINOMA - INFILTRATING LOBULAR',
 'CARCINOMA - MEDULLARY',
 'CARCINOMA - MUCINOUS',
 'CARCINOMA - PAPILLARY - NOS (ADENOCARCINOMA)',
 'CARCINOMA - TUBULAR - INVASIVE CRIBIFORM',
 'INVASIVE MICROPAPILLARY CARCINOMA',
 'METAPLASTIC CARCINOMA',
 'METAPLASTIC CARCINOMA/PSEUDOSARCOMA'
        ]
"""
NYU Dataset report re Ca vs bening:
     
    Benign terms: 
        fibrocystic change (5842), 
        fibroadenoma(3768), 
        hyperplasia (2569), 
        cyst content (2882), 
        benign breast tissue (1364),
        fibrocystic changes (1279), 
        fibrosis (1235), 
        negative for malignancy (1049),
        adipose tissue (1026).
        """
benign_tumours = [
 'ADENOSIS - BLUNT DUCT',
 'ADENOSIS - FLORID',
 'ADENOSIS - NOS',
 'ADENOSIS - SCLEROSING',
 'BENIGN CALCIFICATION',
 'BENIGN SOFT TISSUE TUMOUR',
 'CALCIFICATION',
 'COLUMNAR CELL - ATYPIA - FLAT ATYPIAHYPERPLASIA - INTRADUCTAL - ATYPICAL',
 'COLUMNAR CELL ALTERATION',
 'CYST - NOS',
 'FAT NECROSIS',
 'FIBROADENOMA - NOS',
 'FIBROCYSTIC DISEASE - NOS',
 'FIBROSIS OF THE BREAST',
 'FOREIGN BODY GIANT CELL REACTION',
 'GRANULAR CELL TUMOUR',
 'HAMARTOMA - ADENO - FIBRO - LIPOMA',
 'HYPERPLASIA - LOBULAR - ATYPICAL',
 'HYPERPLASIA DUCTAL',
 'INTRADUCT PAPILLOMATOSIS - ATYPICAL',
 'LOBULAR CARCINOMA IN SITU - LCIS',
 'MUCOCOELE-LIKE LESION',
 'PAPILLOMA - INTRADUCTAL',
 'PAPILLOMATOSIS - INTRADUCTAL - NOS - MULTIPLE',
 'PHYLLODES - BENIGN - NOS'
        ]
"""
NYU Dataset report re Ca vs bening:
     
    Exclusion terms: 
        benign skin (150), 
        explant (93), 
        non-diagnostic (80), 
        no mammary epithelium is identified (59),
        breast capsule (53), 
        breast implant (48), 
        fibrous capsule (48), 
        no benign or malignant epithelial cells seen (48), 
        no mammary epithelial cells (45), 
        dermal scar (43).
        """
exclusions = [
 'ECTASIA - MAMMARY DUCT - PLASMA CELL MASTITIS',
 'GRANULOMATOUS INFLAMMATION',
 'INFLAMMATION - CHRONIC - NOS',
 'MASTITIS',
 'METAPLASIA - APOCRINE',
 'NON-SPECIFIC REACTIVE PROCESS',
 'NORMAL TISSUE - NOS',
 'PHYLLODES - BORDERLINE - NOS',
 'RADIAL SCAR',
 'UNKNOWN'
        ]
