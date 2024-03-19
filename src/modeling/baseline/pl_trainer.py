#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:24:18 2020

@author: mlim-user
"""
import os
import sys
import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from src.modeling.pl_models import SplitBreastModel
from pytorch_lightning.loggers.test_tube import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.constants import DATADIR, NVMEDIR, VALDIR, REPODIR
#%%
class Non_val_epoch_saves(pl.Callback):
    """ custom callback allowing saving of all epochs, regardless of val interval.
    Large images and dataset necessitates speedups wherever possible
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
    
    early_stop_callback = EarlyStopping(
            monitor='avg_val_loss',
            min_delta=0.00,
            patience=10,
            verbose=True,
            mode='min',
            )
    
    iteration = '19May_2GPU_V1'
    print('\n\tIteration/logs name:', iteration)
    
    callback_dir = os.path.join(DATADIR,'dev_test_models/ckpts_' + iteration +'/')
    
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
            save_dir=os.path.join(DATADIR, 'dev_test_logs/tt_logs'),
            name='NYURT_'+iteration,
            description="first fullset on disk",
            debug=False,
            version=None)
    
    trainer = Trainer(
        accumulate_grad_batches=hparams.grad_cum, 
        auto_lr_find=hparams.auto_lr,
        callbacks=[Non_val_epoch_saves(
                iteration=iteration,
                filepath=callback_dir,
                k=1
                )],
        checkpoint_callback=callback,
        check_val_every_n_epoch=hparams.check_val_n,
        default_save_path=callback_dir,
        distributed_backend=hparams.backend,
        early_stop_callback=early_stop_callback, 
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
    
    in_channels = 3 if hparams.use_heatmaps else 1
    model = SplitBreastModel(
            hparams=hparams,
            input_channels=in_channels
            )
    
    if trainer.resume_from_checkpoint is None:
        print('\n\t*** loading original NYU model and weights ***')
        model.load_state_dict(
                torch.load(
                        os.path.join(REPODIR, 'models/sample_imageheatmaps_model.p')
                        )["model"]
                )
                
                #%%
    
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--augmentation', action='store_true', default=True)
    parser.add_argument('--auto_lr', type=bool, default=False)
    parser.add_argument('--backend', type=str, default='ddp')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--check_val_n', type=int, default=2)
    parser.add_argument('--device_type', type=str, default='gpu')
    parser.add_argument('--gpus', default=-1)
    parser.add_argument('--grad_cum', default=6)
    parser.add_argument('--lr', type=float, default=1e-4) # if using cyclic schedule == triangular, bottom of cycle/min
    parser.add_argument('--lr_schedule', type=str, default='ROP')
    parser.add_argument('--max_crop_noise', type=int, default=200)
    parser.add_argument('--max_crop_size_noise', type=int, default=200)
    parser.add_argument('--model_mode', type=str, default='view_split')
    parser.add_argument('--model_path', type=str, default=os.path.join(REPODIR, '/models/sample_imageheatmaps_model.p'))
    parser.add_argument('--max_epochs', default=800)
    parser.add_argument('--num_nodes', default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--overfit_pct', type=float, default=0.06)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--restore', type=str, default=None)
#        default=os.path.join(DATADIR,
#        'dev_test_models/ckpts_30Apr_2GPU_dpp/NYURT_30Apr_2GPU_dpp_NYURT_30Apr_2GPU_dpp__ckpt_epoch_394.ckpt'
#            ))
    parser.add_argument('--seed', type=int, default=22117)
    parser.add_argument('--train_exam_list_fp', type=str, default=os.path.join(NVMEDIR, 'train_ims_master/data_sf2.pkl'))
    parser.add_argument('--train_heatmaps_path', type=str, default=os.path.join(NVMEDIR, 'train_ims_master/heatmaps_sf2'))
    parser.add_argument('--train_image_path', type=str, default=os.path.join(NVMEDIR, 'train_ims_master/cropped_images_sf2'))
    parser.add_argument('--use_heatmaps', action='store_true', default=True)
    parser.add_argument('--use_hdf5', action='store_true', default=True)
    parser.add_argument('--use_n_exams', default=False)
    parser.add_argument('--val_exam_list_fp', type=str, default=os.path.join(NVMEDIR, 'val_ims_master/data_sf2.pkl'))
    parser.add_argument('--val_heatmaps_path', type=str, default=os.path.join(NVMEDIR, 'val_ims_master/heatmaps_sf2'))
    parser.add_argument('--val_image_path', type=str, default=os.path.join(NVMEDIR, 'val_ims_master/cropped_images_sf2'))
    parser.add_argument('--weight_decay', type=float, default=0) #8e-06)

    hparams = parser.parse_args()
    hparams.pl_ver = pl.__version__
    hparams.env = sys.path
    print('\n\tWeight_decay:', hparams.weight_decay)

    main(hparams)
