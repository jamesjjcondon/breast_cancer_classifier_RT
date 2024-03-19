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
import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from src.modeling.pl_models import SplitBreastModel
from pytorch_lightning.loggers.test_tube import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.utilities.tools import compute_preds_acc_no_cpu
from src.constants import DATADIR, NVMEDIR, REPODIR
#%%
class Non_val_epoch_saves(pl.Callback):
    """ custom callback allowing saving of all epochs, regardless of val interval
    large images and dataset necessitates speedups wherever possible
    (validation is very expensive) """
    def __init__(self, iteration, filepath, k):
        self.iteration = iteration
        self.filepath = filepath
        self.k = k
        self.ver = int(self.iteration[-1])
        #if any(self.iteration in x for x in os.listdir(self.filepath)):
        #    self.ver += 1
            
    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.k == 0:
            metrics = trainer.callback_metrics
            if 'avg_val_loss' in metrics:
                avl = metrics['avg_val_loss']
                avl = f'{avl:.3f}'
            else:
                avl = 'NA'
            tl = metrics.get(trainer.checkpoint_callback.monitor)
            current_tl = f'{tl:0.3f}'
            self.name = self.iteration[:-1] + str(self.ver) + '_epo='+ \
                str(trainer.current_epoch) + \
                    '_tloss' + '=' + \
                current_tl + '_' + \
                    'avloss=' + avl + \
                    '.ckpt'
            trainer.checkpoint_callback._save_model(filepath=os.path.join(
                self.filepath, self.name)
                )
        
def main(hparams):
    
    iteration = '01NOV_HMs_rand_init_v1_restart2'
    print('\n\tIteration/logs name:', iteration)
    
    callback_dir = os.path.join(DATADIR,'dev_test_models/3-channel_rand-init/ckpts_' + iteration +'/')
    
    callback = ModelCheckpoint(
            filepath=callback_dir,
            monitor='loss',
            verbose=1,
            save_top_k=-1,
            save_weights_only=False,
            mode='min',
            period=1,
            prefix=iteration + '_'
            )
            
    logger = TestTubeLogger(
            save_dir=os.path.join(DATADIR, 'dev_test_logs/tt_logs/3-channel_rand-init'),
            name='NYURT_'+iteration,
            description="NYU2_HMs_rand_init",
            debug=False,
            version=None)
    
    trainer = Trainer(
        accumulate_grad_batches=hparams.grad_cum, 
        callbacks=[Non_val_epoch_saves(
                iteration=iteration,
                filepath=callback_dir,
                k=1
                )],
        checkpoint_callback=callback,
        check_val_every_n_epoch=hparams.check_val_n,
        default_save_path=callback_dir,
        distributed_backend=hparams.backend,
        early_stop_callback=None, #early_stop_callback, 
        fast_dev_run=False,  
        gpus=hparams.gpus,
        logger=logger,
        max_nb_epochs=hparams.max_epochs, 
        num_nodes=hparams.num_nodes, 
        overfit_pct=hparams.overfit_pct, 
        precision=hparams.precision,
        profiler=False,
        
        resume_from_checkpoint=hparams.restore
            )
    
    model = SplitBreastModel(
            hparams=hparams
            )
    
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
    parser.add_argument('--backend', type=str, default='ddp')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--check_val_n', type=int, default=1)
    parser.add_argument('--device_type', type=str, default='gpu')
    parser.add_argument('--gpus', default=-1)
    parser.add_argument('--grad_cum', default=6)
    parser.add_argument('--lr', type=float, default=1e-08) # if using cyclic schedule == triangular, bottom of cycle/min
    parser.add_argument('--lr_schedule', type=str, default='cyclic')
    parser.add_argument('--max_crop_noise', type=int, default=200)
    parser.add_argument('--max_crop_size_noise', type=int, default=200)
    parser.add_argument('--model_mode', type=str, default='view_split')
    parser.add_argument('--model_path', type=str, default=None) #os.path.join(REPODIR, '/models/sample_imageheatmaps_model.p'))
    parser.add_argument('--max_epochs', default=300)
    parser.add_argument('--num_nodes', default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--overfit_pct', type=float, default=.10)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--rand_init', action='store_true', default=True) #restarted after epoch 3
    parser.add_argument('--restore', type=str, default=None)
    parser.add_argument('--seed', type=int, default=22117)
    parser.add_argument('--train_exam_list_fp', type=str, default=os.path.join(NVMEDIR, 'train_ims_master/small_matched/data_sf_small_NYUC_matched.pkl'))
    parser.add_argument('--train_heatmaps_path', type=str, default=os.path.join(NVMEDIR, 'train_ims_master/small_matched/heatmaps_sf_small_matched'))
    parser.add_argument('--train_image_path', type=str, default=os.path.join(NVMEDIR, 'train_ims_master/small_matched/cropped_ims_NYU_small'))
    parser.add_argument('--use_heatmaps', action='store_true', default=True)
    parser.add_argument('--use_hdf5', action='store_true', default=True)
    parser.add_argument('--use_n_exams', default=False)
    parser.add_argument('--val_exam_list_fp', type=str, default=os.path.join(NVMEDIR, 'val_ims_master/small_matched/data_NYUC_sf_small_matched.pkl'))
    parser.add_argument('--val_heatmaps_path', type=str, default=os.path.join(NVMEDIR, 'val_ims_master/heatmaps_sf_small_matched'))
    parser.add_argument('--val_image_path', type=str, default=os.path.join(NVMEDIR, 'val_ims_master/cropped_ims_NYUC_small'))
    parser.add_argument('--weight_decay', type=float, default=2e-05)

    hparams = parser.parse_args()
    hparams.pl_ver = pl.__version__
    hparams.input_channels = 3 if hparams.use_heatmaps else 1
    hparams.env = sys.path
    print('\n\tWeight_decay:', hparams.weight_decay)

    main(hparams)
