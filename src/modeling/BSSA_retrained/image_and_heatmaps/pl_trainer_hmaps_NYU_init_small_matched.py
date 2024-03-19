#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:24:18 2020

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
import os
import sys
from datetime import datetime
import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from src.modeling.pl_models import SplitBreastModel
from pytorch_lightning.loggers.test_tube import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from src.constants import DATADIR, NVMEDIR, VALDIR, REPODIR
#%%
     
def main(hparams):
    
    early_stop_callback = EarlyStopping(
            monitor='val_loss_epoch',
            min_delta=1,
            patience=4,
            verbose=True,
            mode='min',
            strict=False
            )
        
    date_time = datetime.now().strftime("%Y-%m-%d")
    
    ckpt_callback = ModelCheckpoint(
            filepath=None, #callback_dir,
            monitor='val_loss_epoch',
            verbose=1,
            save_top_k=5,
            save_weights_only=False,
            mode='min',
            period=1,
            filename='{epoch}-{train_loss_epoch:.3f}-{val_loss_epoch:.3f}'
            )
            
    logger = TestTubeLogger(save_dir=os.path.join(DATADIR, 'logs/transfer_learning/NYU2'),
            name='NYURT_'+date_time)
    
    trainer = Trainer(
        accelerator=hparams.accel,
        accumulate_grad_batches=hparams.grad_cum,
        amp_backend='native',
        auto_lr_find=hparams.autolr,
        benchmark=True,
        callbacks=[ckpt_callback], #, early_stop_callback],
        check_val_every_n_epoch=hparams.check_val_n,
        fast_dev_run=False,
        gpus=hparams.gpus,
        #limit_train_batches=6,
        #limit_val_batches=6,
        logger=logger,
        max_epochs=hparams.max_epochs,
        num_nodes=hparams.num_nodes,
        #overfit_batches=12,
        plugins='ddp_sharded',
        precision=hparams.precision,
        profiler=False,
        progress_bar_refresh_rate=10,
        reload_dataloaders_every_epoch=True,
        
        resume_from_checkpoint=hparams.restore,
        
        sync_batchnorm=True,
        #track_grad_norm=2
            )
    
    model = SplitBreastModel(hparams=hparams)
    
    if hparams.rand_init:
        print('\n\t*** randomly initialising weights / training from scratch ***\n')
        
    elif trainer.resume_from_checkpoint is None:
        print('\n\t*** loading original NYU model and weights ***\n')
        model.load_state_dict(
                torch.load(
                        os.path.join(REPODIR, 'models/sample_imageheatmaps_model.p')
                        )["model"]
                )
                
    elif trainer.resume_from_checkpoint is not None:
        print('\n\t*** loaded from checkpoint:', hparams.restore)
    
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--augmentation', action='store_true', default=True)
    parser.add_argument('--autolr', action='store_true', default=False)
    parser.add_argument('--accel', default='ddp_spawn')
    
    parser.add_argument('--train_bsize', type=int, default=6)
    parser.add_argument('--val_bsize', type=int, default=6)

    parser.add_argument('--check_val_n', type=int, default=4)
    parser.add_argument('--device_type', type=str, default='gpu')
    parser.add_argument('--gpus', default=-1)
    parser.add_argument('--grad_cum', default=24)
    parser.add_argument('--lr', type=float, default=0.0007585775750291836) # if using cyclic schedule == triangular, bottom of cycle/min
    parser.add_argument('--lr_schedule', type=str, default='ROP')
    parser.add_argument('--max_crop_noise', type=int, default=100)
    parser.add_argument('--max_crop_size_noise', type=int, default=100)
    parser.add_argument('--model_mode', type=str, default='view_split')
    parser.add_argument('--model_path', type=str, default=os.path.join(REPODIR, '/models/sample_imageheatmaps_model.p'))
    parser.add_argument('--max_epochs', default=2000)
    parser.add_argument('--num_nodes', default=1)
    
    parser.add_argument('--num_workers', type=int, default=0)
    
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--rand_init', action='store_true', default=False)
    parser.add_argument('--restore', type=str, #default=None)
                        default='/data/james/NYU_retrain/logs/transfer_learning/NYU2/NYURT_2021-03-04/version_0/checkpoints/epoch=83-train_loss_epoch=2.030-val_loss_epoch=1.977.ckpt')
   
    parser.add_argument('--seed', type=int, default=22117)
    parser.add_argument('--train_exam_list_fp', type=str, default=os.path.join(NVMEDIR, 'train_ims_master/small_matched/data_sf_small_NYUC_matched.pkl'))
    parser.add_argument('--train_heatmaps_path', type=str, default=os.path.join(NVMEDIR, 'train_ims_master/small_matched/heatmaps_sf_small_matched'))
    parser.add_argument('--train_image_path', type=str, default=os.path.join(NVMEDIR, 'train_ims_master/small_matched/cropped_ims_NYU_small'))
    parser.add_argument('--use_heatmaps', action='store_true', default=True)
    parser.add_argument('--use_hdf5', action='store_true', default=True)
    parser.add_argument('--use_n_exams', default=False)
    parser.add_argument('--val_exam_list_fp', type=str, default=os.path.join(NVMEDIR, 'val_ims_master/small_matched/data_NYUC_sf_small_matched.pkl'))
    parser.add_argument('--val_heatmaps_path', type=str, default=os.path.join(NVMEDIR, 'val_ims_master/small_matched/heatmaps_sf_small_matched'))
    parser.add_argument('--val_image_path', type=str, default=os.path.join(NVMEDIR, 'val_ims_master/small_matched/cropped_ims_NYU_small'))
    parser.add_argument('--weight_decay', type=float, default=4e-06)

    hparams = parser.parse_args()
    # import IPython; IPython.embed()
    if hparams.rand_init:
        assert hparams.restore is None, "\n\n\tEither rand_init or restore from checkpoint, not both."
    hparams.pl_ver = pl.__version__
    hparams.input_channels = 3 if hparams.use_heatmaps else 1
    hparams.env = sys.path
    print('\n\tWeight_decay:', hparams.weight_decay)
    
    main(hparams)
