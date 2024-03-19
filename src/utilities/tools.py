# Copyright (C) 2019 Nan Wu, Jason Phang, Jungkyu Park, Yiqiu Shen, Zhe Huang, Masha Zorin, 
#   Stanisław Jastrzębski, Thibault Févry, Joe Katsnelson, Eric Kim, Stacey Wolfson, Ujas Parikh, 
#   Sushma Gaddam, Leng Leng Young Lin, Kara Ho, Joshua D. Weinstein, Beatriu Reig, Yiming Gao, 
#   Hildegard Toth, Kristine Pysarenko, Alana Lewin, Jiyon Lee, Krystal Airola, Eralda Mema, 
#   Stephanie Chung, Esther Hwang, Naziya Samreen, S. Gene Kim, Laura Heacock, Linda Moy, 
#   Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of breast_cancer_classifier.
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
Defines utility functions for various tasks in breast_cancer_classifier.
"""
import torch
import pandas as pd
import numpy as np
from src.constants import VIEWANGLES


def partition_batch(ls, size):
    """
    Partitions a list into buckets of given maximum length.
    """
    i = 0
    partitioned_lists = []
    while i < len(ls):
        partitioned_lists.append(ls[i: i+size])
        i += size
    return partitioned_lists

def compute_preds_acc_no_cpu(self, output, targets, mode='view_split'):
    assert mode == 'view_split', "Not tested for other modes"
    with torch.no_grad():
        CCys = [targets['L-CC']['benign'], targets['R-CC']['benign'], targets['L-CC']['ca'], targets['R-CC']['ca']]
        MLOys = [targets['L-MLO']['benign'], targets['R-MLO']['benign'], targets['L-MLO']['ca'], targets['R-MLO']['ca']]
        import IPython; IPython.embed()

        total = torch.tensor(0, dtype=torch.float, device=self.device)
        correct = torch.tensor(0, dtype=torch.float, device=self.device)
        CC_yhat, MLO_yhat = output.values()
        for view, y_hats, ys in zip(VIEWANGLES.LIST, [CC_yhat, MLO_yhat], [CCys, MLOys]):
            for i in range(4):
                max_logit, class_pred = torch.max(output[view][:,i], dim=1) # returns logit and index of max value
                # y_hats.append(ix) # so order of view-wise yhats is same as output == Left benign, R benign, L ca, R ca
                total += torch.tensor(len(ys[i]), dtype=torch.float)
                correct += (ys[i] == y_hats[i]).sum().float()
    return correct / total
    

def applyLUT_and_window_exp(dcm, maxv=65535, window='Normal'):
    """ 
    implemented from https://dicom.innolitics.com/ciods/digital-mammography-x-ray-image/voi-lut/00281056 
    
    # maxv determines max value and histogram of output image. 
    Must be 65535 (max possible value for np.uint16 / 16bit values)
    for saving to png with PIL / Image.fromarray()
        otherwise ouput is way too dark.
        
    # with native data types, LUT interpolates highest density tissues through upper pixel range into low density
    (ie distorts / ruins density representation)
    
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
            - # changed from dcm.BitsStored
    :: Returns
        :: image only
    """
    assert dcm.PresentationLUTShape == 'INVERSE', dcm.PhotometricInterpretation == 'MONOCHROME1'
    image = dcm.pixel_array.astype(np.int32) #
    
    print(dcm.Manufacturer)
    # store window centres and widths:
    win = pd.DataFrame([list(dcm.WindowCenter), list(dcm.WindowWidth)], index=['dcm.WindowCenter', 'dcm.WindowWidth'], columns=list(dcm.WindowCenterWidthExplanation))
    WC = int(np.round(win.loc['dcm.WindowCenter',window]))
    WW = int(np.round(win.loc['dcm.WindowWidth',window]))
    
    # apply sigmoid LUT function:    
    image = (4 * ((image - WC) / WW)) # positive 4 (versus -4) inverts
    
    image = 65535 / (1 + (np.exp(image)))
    
    return image.astype(np.uint32) #.ast
