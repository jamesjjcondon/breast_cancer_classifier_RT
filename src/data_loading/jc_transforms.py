#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 18:08:48 2020

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

from __future__ import division
import torch
import random
import numpy as np
import cv2
from skimage.transform import resize, rotate
from skimage.exposure import equalize_adapthist as clahe

class CLAHE(object):
    """ Uses scikit-image.exposure
    Contrast Limited Adjusted Histogram Equialisation. 
    See
        Pisano 1998 - Contrast limited adaptive histogram equalization image processing to improve the detection of simulated spiculations in dense mammograms.
        Teare 2017 - Malignancy Detection on Mammography Using Dual Deep Convolutional Neural Networks and Genetically Discovered False Color Input Enhancement
            https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5537100/

    Args:
        clip_limit: – Threshold for contrast limiting.
        tile_grid = Size of grid for histogram equalization. Input image will be divided into equally sized rectangular tiles 
    """
    def __init__(self, clip_limit, nbins):
        self.clip_limit = clip_limit # 0.045 
        self.nbins = nbins # 75
        
    def __call__(self, img):
        """
        Args:
            np.array : Image to undergoe CLAHE.

        """
        img = (img - img.min()) / (img.max() - img.min())
        out = clahe(img, kernel_size=None, clip_limit=self.clip_limit, nbins=self.nbins)
        return out

    def __repr__(self):
        return self.__class__.__name__ + '()'

class RandomCLAHE(object):
    """ Uses skimage.exposure
    Contrast Limited Adjusted Histogram Equialisation. 
    See
        Pisano 1998 - Contrast limited adaptive histogram equalization image processing to improve the detection of simulated spiculations in dense mammograms.
        Teare 2017 - Malignancy Detection on Mammography Using Dual Deep Convolutional Neural Networks and Genetically Discovered False Color Input Enhancement
            https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5537100/

    Args:
        clip_limit: – Threshold for contrast limiting.
        tile_grid = Size of grid for histogram equalization. Input image will be divided into equally sized rectangular tiles 
    """
    def __init__(self, clip_limit, nbins, p):
        self.p = p
        self.clip_limit = clip_limit # 0.045 
        self.nbins = nbins # 75
        
    def __call__(self, img):
        """
        Args:
            np.array : Image to undergoe CLAHE.

        """
        img = (img - img.min()) / (img.max() - img.min())
        if random.random() < self.p:
            out = clahe(img, kernel_size=None, clip_limit=self.clip_limit, nbins=self.nbins)
            return out
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
#class cvCLAHE(object):
#    """ Uses cv2.createCLAHE
#    Contrast Limited Adjusted Histogram Equialisation. 
#    See
#        Pisano 1998 - Contrast limited adaptive histogram equalization image processing to improve the detection of simulated spiculations in dense mammograms.
#        Teare 2017 - Malignancy Detection on Mammography Using Dual Deep Convolutional Neural Networks and Genetically Discovered False Color Input Enhancement
#            https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5537100/
#        cv2 docs: https://docs.opencv.org/3.0-beta/modules/cudaimgproc/doc/histogram.html
#
#    Args:
#        clip_limit: – Threshold for contrast limiting.
#        tile_grid = Size of grid for histogram equalization. Input image will be divided into equally sized rectangular tiles 
#    """
#    def __init__(self, clip_limit, tile_grid):
#        self.clip_limit = clip_limit
#        self.tile_grid = tile_grid
#        
#    def __call__(self, img):
#        """
#        Args:
#            np.array : Image to undergoe CLAHE.
#
#        """
#        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid)
#        clahe_img = clahe.apply(img)
#        return clahe_img
#
#    def __repr__(self):
#        return self.__class__.__name__ + '()'
    
class int32_Invert(object):
    """Uses np.invert() to convert values of array.
    Effectively inverts contrast and changes dicoms from Monochrome2 to Monochrome1

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __call__(self, img):
        """
        Args:
            np.array : Image to be flipped.

        Returns:
            np.array: Randomly flipped.
        """
        img = np.int32(img)
        return np.invert(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class m0_v1_norm(object):
    """Normalises a 2d array to a mean of zero and a std / variance of one.
    """

    def __call__(self, array):
        """
        Args:
            array to be converted.

        Returns:
            normalised array.
        """
        array = ((array - array.mean()) / (array.std()))
#        pic = np.int32(pic)
        
        return array

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class Min_max_norm(object):
    """Normalises a 2d array to between zero and one.
    """

    def __call__(self, array):
        """
        Args:
            array to be converted.

        Returns:
            normalised array.
        """
        array = ((array - array.min()) / (array.max() - array.min()))
#        pic = np.int32(pic)
        
        return array

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class RandomHorizontalFlip(object):
    """Horizontally flip the given np.array randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            np.array : Image to be flipped.

        Returns:
            np.array: Randomly flipped.
        """
        if random.random() < self.p:
            return cv2.flip(img, 1)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
class RandomRotate(object):
    """Rotate the image by angle.

  
    Rotate the image at a given probability, from a degree randomly chosen by a user, specified range
    
    Args:
        p (float): probability of the image being rotated. Default value is 0.5
        rrange (tuple): limits of possible rotation 
    """

    def __init__(self, p=0.5, rrange=(0,360)):
        self.p = p
        self.rrange = rrange

    def __call__(self, img):
        """
        Args:
            np.array : Image to be rotated.

        Returns:
            np.array: Randomly rotated by probability p.
        """
        if (self.rrange[0] and self.rrange[1]) >= 0:
            deg = random.choice(range(self.rrange[0], self.rrange[1]))
        
        elif self.rrange[0] < 0:
            self.lb = self.rrange[0] + 360
            tot_range = list(range(self.lb+1,361)) + list(range(1,self.rrange[1]+1))
            deg = random.choice(tot_range)
        
        if random.random() < self.p:
            img = rotate(img, deg, resize=False, 
                   order=3, # bicubic interpolation 
                   mode='constant',
                   cval=img.min(), # pad with min val if needed.
                   preserve_range=True) # don't normalise yet
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    """Vertically flip the given np.array randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            np.array : Image to be flipped.

        Returns:
            np.array: Randomly flipped.
        """
        if random.random() < self.p:
            return cv2.flip(img, 0)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
class Resize(object):
    """Resize the input array to the given size.
    
    Review
        # https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize
        and
        # https://scikit-image.org/docs/dev/api/skimage.transform.html?highlight=warp#warp
        For important info about the effect of this function on:
            - pixel_value ranges
            - normalisation and
            - Interpolation
            
    height : width in ds == 1.089522
    width = height / 1.089522
    CURRENTLY CONVERTS FROM UINT16 TO NUMPY.FLOAT64
    """    
    

    def __init__(self, size):
        assert (isinstance(size, int) and len(size) == 2) or isinstance(size, float) or isinstance(size, tuple)
        self.scaleto = size

    def __call__(self, img):
        """
        Args:
            img : array to be scaled.

        Returns:
            : Rescaled array.
        """
        if self.scaleto is not None:
                if type(self.scaleto) is float:
                    self.nh = np.round(self.scaleto * img.shape[0]) # new hight
                    self.nw = np.round(self.scaleto * img.shape[1]) # new width
                elif type(self.scaleto) is tuple:
                    self.nh = self.scaleto[0]
                    self.nw = self.scaleto[1]
                elif self.scaleto == 'Wu_NYU_2019':
                    if self.view == 'CC':
                        self.nh = 2677
                        self.nw = 1942
                    if self.view == 'MLO':
                        self.nh = 2974
                        self.nw = 1748
#        print('\nResizing to', self.nh, ' x ', self.nw)
       
        img = resize(img, (self.nh, self.nw), 
                    order=3, # bicubic interpolation
                    mode='constant', 
                    clip=True,
                    preserve_range=True) # does not normalise
        # To do: check precision of skimage.resize
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.scaleto)
    
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
#        return F.normalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class ToTensor(object):
    """Convert a 2d array of type uint16 to a 3d torch tensor.
    Uses np.newaxis to add arbitrary 3rd axis.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        pic = pic[np.newaxis, :, :]
#        pic = np.int32(pic)
        
        return torch.from_numpy(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ds_m_unit_v_norm(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean and std for whole ds, transform
    will normalize 

    Args:
        mean of ds
        std of ds
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, array):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        array -= self.mean
        array /= self.std
        return array

    def __repr__(self):
        return self.__class__.__name__ + '()'
