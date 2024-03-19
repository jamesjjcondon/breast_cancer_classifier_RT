#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
Defines utility functions for reading png and hdf5 images.
"""
import numpy as np
import pandas as pd
import imageio
import h5py
import pydicom

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
    """ push min to 0 for NYU cropping and ideal centre pipeline: """
    image -= image.min()
#    print(image.min(), image.max())
    return image.astype(np.uint16)


def read_image_png(file_name):
    image = np.array(imageio.imread(file_name))
    return image

def read_image_dcm(file_name, window='Normal', return_dcm=False):
    """
    Loads image from dicom
    If PresentationLUTShape is 'INVERSE', applies function to revert
    Standardises to zero minimum.
    """
    dcm = pydicom.read_file(file_name)
    if dcm.PresentationLUTShape == 'INVERSE':
        image = applyLUT_and_window_exp2(
                        dcm,
                        window=window)
    else:
        image = dcm.pixel_array
#    print('image type', image.type())
    if return_dcm:
        return image, dcm
    else:
        return image

def read_image_mat(file_name): #, hm=False):
    data = h5py.File(file_name, 'r')
#    if hm:
#        name = 'hm_image'
#    else:
#        name = 'image'
    try:
        image = np.array(data['image']).T
    except:
        image = np.array(data['hm_image']).T
    data.close()
    return image

def read_image_and_attrs_mat(file_name):
    """ Reads and returns both image and attributes from hdf5"""
    data = h5py.File(file_name, 'r')
    try:
        image = np.array(data['image']).T
    except:
        image = np.array(data['hm_image']).T
    try:
        attrs = dict(data['image'].attrs.items())
    except:
        attrs = dict(data['hm_image'].attrs.items())
    data.close()
    return image, attrs
