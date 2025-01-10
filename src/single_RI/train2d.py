import argparse
import os
import torch
from data_module import MoleculeDataModuleSplitted
from model import GCN
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from hparam import Hyperparameters

def train_model(check_mode, gpu_id=None):
    print('Running in check mode:', check_mode)
    
    if gpu_id is not None:
        if gpu_id >= torch.cuda.device_count():
            raise ValueError(f"GPU {gpu_id} is not available. Available GPUs: {torch.cuda.device_count()}")
        
        print(f'Using GPU: {gpu_id}')
        devices = [gpu_id]
    else:
        print('Using automatic device selection')
        devices = 'auto'
    
    print('Training the GCN model...')

    # Define paths
    train_dir = 'data/2D_AKPEG/data-GCxGC_train.csv'
    val_dir = 'data/2D_AKPEG/data-GCxGC_val.csv'
    test_dir = 'data/2D_AKPEG/data-GCxGC_test.csv'
    # Initialize the data module
    data_module = MoleculeDataModuleSplitted(train_path=train_dir,
                                            val_path=val_dir,
                                            test_path=test_dir,
                                            batch_size=Hyperparameters.batch_size,
                                            num_workers=Hyperparameters.num_workers,
                                            random_state=Hyperparameters.random_state,
                                            ri2d=True)

    # Initialize the model
    model = GCN(num_node_features=-1, 
                hidden_channels=Hyperparameters.hidden_channels, 
                learning_rate=Hyperparameters.learning_rate,
                weight_decay=Hyperparameters.weight_decay,
                lr_factor=Hyperparameters.lr_factor,
                lr_patience=Hyperparameters.lr_patience,
                dropout=Hyperparameters.dropout,
                n_tasks=2)  

    # Set up callbacks and logger
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    wandb_logger = WandbLogger(project='GCN-Molecule', save_dir='wandb_log')
    check_point = ModelCheckpoint(monitor='val_loss')

    # Initialize the trainer
    trainer = Trainer(devices=devices,
                      accelerator='auto',
                      max_epochs=Hyperparameters.max_epoch,
                      min_epochs=Hyperparameters.min_epoch,
                      logger=wandb_logger,
                      callbacks=[lr_monitor, check_point],
                      fast_dev_run=check_mode,
                      log_every_n_steps=300)

    # Train and test the model
    trainer.fit(model, datamodule=data_module)
    trainer.validate(model, datamodule=data_module, ckpt_path='best')
    # trainer.test(model, datamodule=data_module, ckpt_path='best')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the GCN model')
    parser.add_argument('--check', action='store_true', help='Run in check mode (fast dev run)')
    parser.add_argument('--gpu', type=int, help='Specific GPU ID to use (optional)')
    args = parser.parse_args()

    train_model(check_mode=args.check, gpu_id=args.gpu)