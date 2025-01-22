import argparse
import os
import torch
from data_module import MoleculeDataModule
from model import GCN
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from hparam import Hyperparameters
torch.set_float32_matmul_precision('medium')

check_mode = Hyperparameters.check_mode
print('Running in check mode:', check_mode)
print('Training the GCN model...')

# Define paths
data_dir = 'data/P_VAN_RAMP/P-VAN-RAMP-G-C-P.csv'  

# Initialize the data module
data_module = MoleculeDataModule(csv_path=data_dir,
                                    batch_size=Hyperparameters.batch_size,
                                    num_workers=Hyperparameters.num_workers,
                                    test_size=Hyperparameters.test_size,
                                    random_state=Hyperparameters.random_state)

# Initialize the model
model = GCN(num_node_features=-1, 
            hidden_channels=Hyperparameters.hidden_channels, 
            learning_rate=Hyperparameters.learning_rate,
            weight_decay=Hyperparameters.weight_decay,
            lr_factor=Hyperparameters.lr_factor,
            lr_patience=Hyperparameters.lr_patience,
            dropout=Hyperparameters.dropout)  

# Set up callbacks and logger
lr_monitor = LearningRateMonitor(logging_interval='epoch')
wandb_logger = WandbLogger(project='GCRI-GCN', save_dir='wandb_log')
check_point = ModelCheckpoint(monitor='val_loss')

# Initialize the trainer
trainer = Trainer(devices='auto',
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

