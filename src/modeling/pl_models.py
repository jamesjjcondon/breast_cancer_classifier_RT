#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:22:52 2020

@author: James Condon
"""
# Modified from and based on:

# Copyright (C) 2019 Nan Wu, Jason Phang, Jungkyu Park, Yiqiu Shen, Zhe Huang, Masha Zorin, 
#   Stanisław Jastrzębski, Thibault Févry, Joe Katsnelson, Eric Kim, Stacey Wolfson, Ujas Parikh, 
#   Sushma Gaddam, Leng Leng Young Lin, Kara Ho, Joshua D. Weinstein, Beatriu Reig, Yiming Gao, 
#   Hildegard Toth, Kristine Pysarenko, Alana Lewin, Jiyon Lee, Krystal Airola, Eralda Mema, 
#   Stephanie Chung, Esther Hwang, Naziya Samreen, S. Gene Kim, Laura Heacock, Linda Moy, 
#   Kyunghyun Cho, Krzysztof J. Geras
#
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


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from src.data_loading.drjc_datasets import BSSA_exams
from torch.optim import Adam
import src.modeling.layers as layers
from src.constants import VIEWS, VIEWANGLES
import gc



class SplitBreastModel(pl.LightningModule):
    """ Best-model for malignacy detection in both screening and Bx subpop
    'view-wise' in paper - Fig5.a 
    Forward pass concatenates:
        - L and R CC
            and
        - L and R MLO
    Passes ft vecs from each separately through fc layers
    Each generates L and R benign and malignant probabilities
        == 8 scores (LCC-ca, LCC-ben, RCC-ca, RCC-ben ... RMLO-ben.)
    Left and right breast malignant and benign scores are averaged from 2 heads.
    """
    def __init__(self, hparams):
        super(SplitBreastModel, self).__init__()
        """ """
        self.hparams = hparams
        self.train_trsfm = transforms.Compose([
#            jc_transforms.RandomCLAHE(clip_limit=0.045, nbins=75, p=0.5),
            transforms.ToTensor(),
        ])
        try:
            input_channels = 3 if hparams.use_heatmaps else 1
        except:
            input_channels = 3 if hparams['use_heatmaps'] else 1

        self.four_view_resnet = FourViewResNet(input_channels)

        self.fc1_cc = nn.Linear(256 * 2, 256 * 2)
        self.fc1_mlo = nn.Linear(256 * 2, 256 * 2)
        self.output_layer_cc = layers.OutputLayer(256 * 2, (4, 2), log_softmax=True)
        self.output_layer_mlo = layers.OutputLayer(256 * 2, (4, 2), log_softmax=True)

        self.all_views_avg_pool = layers.AllViewsAvgPool()
        self.all_views_gaussian_noise_layer = layers.AllViewsGaussianNoise(0.01)

        self.accuracy = pl.metrics.Accuracy()
        
    def forward(self, x):
        h = self.all_views_gaussian_noise_layer(x)
        result = self.four_view_resnet(h)
        h = self.all_views_avg_pool(result)

        # Pool, flatten, and fully connected layers
        h_cc = torch.cat([h[VIEWS.L_CC], h[VIEWS.R_CC]], dim=1) # CCs concatenated """
        h_mlo = torch.cat([h[VIEWS.L_MLO], h[VIEWS.R_MLO]], dim=1) # MLOs concatenated """

        h_cc = F.relu(self.fc1_cc(h_cc))
        h_mlo = F.relu(self.fc1_mlo(h_mlo))
        
        h_cc = self.output_layer_cc(h_cc)
        h_mlo = self.output_layer_mlo(h_mlo)

        h = {
            VIEWANGLES.CC: h_cc,
            VIEWANGLES.MLO: h_mlo,
        }
        
        return h
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        train_loss = self.loss(out, y)
        #batch_accuracy = self.compute_preds_acc_no_cpu(out, y, self.hparams.train_bsize)
        self.log('train_loss', train_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        #self.log('train_acc', batch_accuracy, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return train_loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        val_loss = self.loss(out, y)
        #batch_accuracy = self.compute_preds_acc_no_cpu(out, y, self.hparams.val_bsize)
        self.log('val_loss', val_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        #self.log('val_acc', batch_accuracy, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        gc.collect() #pytorch/issues/40911
        return val_loss
    
    def loss(self, out, y):
        # sums negative log likelihood loss for each view, for each breast for benign and malignant labels
        lossLCCb = F.nll_loss(out['CC'][:,0], y['L-CC']['benign'])
        lossRCCb = F.nll_loss(out['CC'][:,1], y['R-CC']['benign'])
        lossLCCca = F.nll_loss(out['CC'][:,2], y['L-CC']['ca']) 
        lossRCCca = F.nll_loss(out['CC'][:,3], y['R-CC']['ca'])
        lossLMLOb = F.nll_loss(out['MLO'][:,0], y['L-MLO']['benign'])
        lossRMLOb = F.nll_loss(out['MLO'][:,1], y['R-MLO']['benign'])
        lossLMLOca = F.nll_loss(out['MLO'][:,2], y['L-MLO']['ca'])
        lossRMLOca = F.nll_loss(out['MLO'][:,3], y['R-MLO']['ca'])

        lossCa = lossLCCca + lossRCCca + lossLMLOca + lossRMLOca # cancer loss
        lossb = lossLCCb + lossRCCb + lossLMLOb + lossRMLOb # benign loss
        
        return lossCa + lossb
    
    def compute_preds_acc_no_cpu(self, output, targets, batch_size, mode='view_split'):
        assert mode == 'view_split', "Not tested for other modes"
        CCys = [targets['L-CC']['benign'], targets['R-CC']['benign'], targets['L-CC']['ca'], targets['R-CC']['ca']]
        MLOys = [targets['L-MLO']['benign'], targets['R-MLO']['benign'], targets['L-MLO']['ca'], targets['R-MLO']['ca']]
        
        CC_Ys_left_benign_right_benign_left_ca_right_ca = torch.cat(CCys)
        MLO_Ys_left_benign_right_benign_left_ca_right_ca = torch.cat(MLOys)
        concat_targets = torch.cat((CC_Ys_left_benign_right_benign_left_ca_right_ca, MLO_Ys_left_benign_right_benign_left_ca_right_ca))
        
        CC_yhat, MLO_yhat = output.values()
        
        y_hats = torch.cat((CC_yhat, MLO_yhat))
        max_log_softmax, class_preds = torch.max(y_hats, dim=-1)
        
        class_preds = class_preds.view(8*batch_size)

        accuracy = self.accuracy(class_preds, concat_targets)
        return accuracy
    
    def configure_optimizers(self):
        optimiser = Adam(
                self.parameters(), lr=self.hparams.lr,
                betas=(0.9, 0.999), eps=1e-8, 
                weight_decay=self.hparams.weight_decay,
                amsgrad=False
                )

        if self.hparams.lr_schedule == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                    optimiser, 
                    step_size=2,
                    gamma=.2)
            
            return [optimiser], [scheduler]
        
        elif self.hparams.lr_schedule == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimiser,
                base_lr=self.hparams.lr,
                max_lr=0.001,
                step_size_up=3, 
                step_size_down=None, 
                mode='triangular', 
                gamma=0.75,
                scale_fn=None, 
                scale_mode='cycle',
                cycle_momentum=False, 
                base_momentum=0.8, 
                max_momentum=0.9, 
                last_epoch=-1
                )
            return [optimiser], [scheduler]
        
        elif self.hparams.lr_schedule == 'ROP':       
            scheduler = {
                            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                                    optimiser,
                                    mode='min',
                                    factor=0.1, 
                                    patience=2, 
                                    verbose=True,
                                    threshold=1e-04,
                                    threshold_mode='rel',
                                    cooldown=0, 
                                    min_lr=1e-12,
                                    eps=1e-13),
                            'monitor': 'train_loss_epoch'
                            }
            return [optimiser], [scheduler]
        
        else:
            return [optimiser]

    def train_dataloader(self):
        ds = BSSA_exams(
                exam_list_in=self.hparams.train_exam_list_fp,
                transform=self.train_trsfm,
                parameters=self.hparams,
                train_val_or_test='train'
                )

        dloader = DataLoader(
                dataset=ds,
                batch_size=self.hparams.train_bsize,
                num_workers=self.hparams.num_workers,
                pin_memory=True                
                )
        return dloader

    def val_dataloader(self):
        ds = BSSA_exams(
                exam_list_in=self.hparams.val_exam_list_fp,
                transform=self.train_trsfm,
                parameters=self.hparams,
                train_val_or_test='val'
                )
        
        dloader = DataLoader(
                dataset=ds,
                batch_size=self.hparams.val_bsize,
                num_workers=self.hparams.num_workers,
                pin_memory=True, 
                )
        return dloader

     
class FourViewResNet(nn.Module):
    def __init__(self, input_channels):
        super(FourViewResNet, self).__init__()

        self.cc = resnet22(input_channels)
        self.mlo = resnet22(input_channels)
        self.model_dict = {}
        self.model_dict[VIEWS.L_CC] = self.l_cc = self.cc
        self.model_dict[VIEWS.L_MLO] = self.l_mlo = self.mlo
        self.model_dict[VIEWS.R_CC] = self.r_cc = self.cc
        self.model_dict[VIEWS.R_MLO] = self.r_mlo = self.mlo

    def forward(self, x):
        h_dict = {
            view: self.single_forward(x[view], view)
            for view in VIEWS.LIST
        }
        return h_dict

    def single_forward(self, single_x, view):
        return self.model_dict[view](single_x)


class ViewResNetV2(nn.Module):
    """
    Adapted fom torchvision ResNet, converted to v2
    """
    def __init__(self,
                 input_channels, num_filters,
                 first_layer_kernel_size, first_layer_conv_stride,
                 blocks_per_layer_list, block_strides_list, block_fn,
                 first_layer_padding=0,
                 first_pool_size=None, first_pool_stride=None, first_pool_padding=0,
                 growth_factor=2):
        super(ViewResNetV2, self).__init__()
        self.first_conv = nn.Conv2d(
            in_channels=input_channels, out_channels=num_filters,
            kernel_size=first_layer_kernel_size,
            stride=first_layer_conv_stride,
            padding=first_layer_padding,
            bias=False,
        )
        self.first_pool = nn.MaxPool2d(
            kernel_size=first_pool_size,
            stride=first_pool_stride,
            padding=first_pool_padding,
        )

        self.layer_list = nn.ModuleList()
        current_num_filters = num_filters
        self.inplanes = num_filters
        for i, (num_blocks, stride) in enumerate(zip(
                blocks_per_layer_list, block_strides_list)):
            self.layer_list.append(self._make_layer(
                block=block_fn,
                planes=current_num_filters,
                blocks=num_blocks,
                stride=stride,
            ))
            current_num_filters *= growth_factor
        self.final_bn = nn.BatchNorm2d(
            current_num_filters // growth_factor * block_fn.expansion,
        )
        self.relu = nn.ReLU()

        # Expose attributes for downstream dimension computation
        self.num_filters = num_filters
        self.growth_factor = growth_factor

    def forward(self, x):
        h = self.first_conv(x)
        h = self.first_pool(h)
        for i, layer in enumerate(self.layer_list):
            h = layer(h)
        h = self.final_bn(h)
        h = self.relu(h)
        return h

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
        )

        layers_ = [
            block(self.inplanes, planes, stride, downsample)
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers_.append(block(self.inplanes, planes))

        return nn.Sequential(*layers_)
    
def resnet22(input_channels):
    return ViewResNetV2(
        input_channels=input_channels,
        num_filters=16,
        first_layer_kernel_size=7,
        first_layer_conv_stride=2,
        blocks_per_layer_list=[2, 2, 2, 2, 2],
        block_strides_list=[1, 2, 2, 2, 2],
        block_fn=layers.BasicBlockV2,
        first_layer_padding=0,
        first_pool_size=3,
        first_pool_stride=2,
        first_pool_padding=0,
        growth_factor=2
    )
