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
Defines utility functions for saving png and hdf5 images.
"""
import imageio
import h5py
import png
import pydicom
import pandas as pd
from src.utilities.tools import applyLUT_and_window_exp


def save_image_as_png(image, filename):
    """
    Saves image as png files while preserving bit depth of the image
    """
    imageio.imwrite(filename, image)


def save_image_as_hdf5(image, filename):
    """
    Saves image as hdf5 files to preserve the floating point values.
    """
    assert 'p' in filename
    h5f = h5py.File(filename, 'w')
    try:
        h5f.create_dataset(
                'image', 
                data=image.transpose(), #data transposed
                compression=0 #"lzf"
                )
    except:
        print('save_image_as_hdf5('+filename+') did not transpose\n')
        h5f.create_dataset(
                'image', 
                data=image,
                compression=0 #"lzf"
                )
    
#    y = int(filename.rsplit('i',1)[0])
    h5f.close()
    
def save_image_and_attrs_hdf5(image, filename, data):
    """
    Saves image and attributes as hdf5 file to preserve the floating point values.
    # colum names have old suffix eg 'HIST_OUTCOME.string()' - split out
    # if data is one element, only it is stored
    # otherwise store whole array.
    """
    assert 'p' in filename
    assert isinstance(data, pd.core.frame.DataFrame)
    h5f = h5py.File(filename, 'w')
    ds = h5f.create_dataset(
            'image', 
            data=image.transpose(), # why is data transposed?
            compression=0 #"lzf"
            )
    for col in data.columns:
#        print(col) 
        if len(data[col].values) == 1:
            ds.attrs[col.split('.',1)[0]] = data[col].values[0]
            
        elif data[col.split('.',1)[0]].values > 1:
            ds.attrs[col] = data[col].values
            
        else:
            raise ValueError("Saving hdf5 attributes didn't work for {}".format(filename))
    h5f.close()
        
def save_dicom_image_as_png(dicom_filename, png_filename, bitdepth=16):
    """
    Save x-bit mammogram from dicom as rescaled 16-bit png file.
    :param dicom_filename: path to input dicom file.
    :param png_filename: path to output png file.
    :param bitdepth: bit depth of the input image. Set it to 12 for 12-bit mammograms.
    """
    dcm = pydicom.read_file(dicom_filename)
    
    image = applyLUT_and_window_exp(dcm)
    
    with open(png_filename, 'wb') as f:
        writer = png.Writer(height=image.shape[0], width=image.shape[1], bitdepth=bitdepth, greyscale=True)
        writer.write(f, image.tolist())
