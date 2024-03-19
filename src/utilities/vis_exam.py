#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:10:49 2020

@author: james
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import subprocess
import gc
import pydicom
import matplotlib.pyplot as plt
from src.utilities.reading_images import read_image_dcm
from src.constants import CSVDIR

from IPython import embed

class examine():
    def __init__(self, AN, folder, figsize=(20,20), gc=False):
        assert isinstance(AN, int)
        assert folder.startswith('/')
        self.AN = 'A'+str(AN)
        self.folder = folder
        self.gc = gc
        self.figsize=figsize
        
    def AN_to_is_dir(self):
        """ Converts Accession number to intersect filepaths
        input - accession number
        output - intersect filepaths """
        pt = pd.read_csv(os.path.join(CSVDIR, 'dcm_master_info_final.csv'))
        pt = pt[pt.AN == self.AN]
        # embed()
        self.fps = pt.full_fp.values
        del pt

    def download(self):
        """ for arbitrary n of filepaths, downloads from intersect """
        print('\ndownloading dicoms...')
        assert isinstance(self.fps, np.ndarray)
        self.local_fps = []
        for fp in self.fps:
            filename = os.path.split(fp)[-1]
            self.local_fps.append(os.path.join(self.folder, filename))
#            call = 
#            sshpem = "ssh -i /home/mlim-user/.ssh/mlim-user1"
            # rsync -uvh --progress a1142546@samlim-db.uoa.intersect.org.au:/data/samlim-db/bssa/images/ML_29-31_files/I29451-E13-A1536969-S-i1.dcm /home/mlim-user/Downloads/
#            print('subprocess.call(["rsync", "-ruvh", "--progress", "-e"')
#            print(sshpem, "a1142546@samlim-db.uoa.intersect.org.au:/"+fp, self.folder)
            print("\nCall in terminal: \n\t rsync -ruvvh --progress -e 'ssh -i /home/mlim-user/.ssh/mlim-user1' a1142546@samlim-db.uoa.intersect.org.au:/" + fp, self.folder)
        print('\n\t then press enter...')
        input()
        
    def arrs_views_lat(self):
        """ stores arrays, views and lateralities """
        self.LMLO, self.RMLO, self.LCC, self.RCC = [], [], [], []
        for file in tqdm(self.local_fps, 'reading dicoms...'):
            image, dcm = read_image_dcm(file, return_dcm=True)
#            print(dcm.top)
#            plt.figure(figsize=(10,10))
#            plt.imshow(image, cmap='gray')
#            plt.show()
#            input()
            lat = dcm.ImageLaterality
            view = dcm.ViewPosition
            if lat == 'L' and view == 'MLO':
                self.LMLO.append({'view':'LMLO','time':int(dcm.AcquisitionTime), 'arr':image}) #dcm.pixel_array})
            elif lat == 'L' and view == 'CC':
                self.LCC.append({'view':'LCC','time':int(dcm.AcquisitionTime), 'arr':image}) #dcm.pixel_array})
            elif lat == 'R' and view == 'MLO':
                self.RMLO.append({'view':'RMLO','time':int(dcm.AcquisitionTime), 'arr':image}) #dcm.pixel_array})
            elif lat == 'R' and view == 'CC':
                self.RCC.append({'view':'RCC','time':int(dcm.AcquisitionTime), 'arr':image}) #dcm.pixel_array})
            if gc == True:
                del dcm
        self.im_lists = [self.LMLO, self.RMLO, self.LCC, self.RCC]
        if gc == True:
            gc.collect()

    def choose_dups(self):
        """ where >1 im / view (duplicates), choose last 
        !!! Assumes the best image was taken later !!! """
        self.im_count = 0
        self.new_list = []
        for im in self.im_lists:
            self.im_count += len(im)
            if len(im) > 1:
                times = []
                for exam in im:
                    times.append(exam['time'])
                last_ix = times.index(max(times)) # get index of latest acquisition
                self.new_list.append(im[last_ix]) # use only that
            else:
                self.new_list.append(im[0])
        self.LMLO = [x for x in self.new_list if x['view'] == 'LMLO'][0]
        self.LCC = [x for x in self.new_list if x['view'] == 'LCC'][0]
        self.RMLO = [x for x in self.new_list if x['view'] == 'RMLO'][0]
        self.RCC = [x for x in self.new_list if x['view'] == 'RCC'][0]

    def plot(self):
        fig, ax = plt.subplots(2,2, figsize=self.figsize, squeeze=True) #, sharex=True, sharey=True)
        ax[0,1].imshow(self.LMLO['arr'], cmap='gray')
        ax[0,1].set_xticklabels([])
        ax[0,1].set_yticklabels([])
        ax[0,0].imshow(self.RMLO['arr'], cmap='gray')
        ax[0,0].set_xticklabels([])
        ax[0,0].set_yticklabels([])
        ax[1,1].imshow(self.LCC['arr'], cmap='gray')
        ax[1,1].set_xticklabels([])
        ax[1,1].set_yticklabels([])
        
        ax[1,0].imshow(self.RCC['arr'], cmap='gray')
        ax[1,0].set_xticklabels([])
        ax[1,0].set_yticklabels([])
        
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
#        plt.set_yticklabels([])
#        plt.set_xticklabels([])
        title = self.AN + '\nims for this pt: '+str(self.im_count)
        fig.suptitle(title)
        fig.subplots_adjust(wspace=0, hspace=0) #fig.squeeze() #tight_layout()
        plt.show()

def vis_exam(AN, folder):
    """ input: accession number
    downloads from intersect
    visualises all exams for that AN """
    inst = examine(AN, folder)
    inst.AN_to_is_dir()
    inst.download()
    inst.arrs_views_lat()
    inst.choose_dups()
    inst.plot()
    return inst
