# model.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import pytorch_lightning as pl
import torch.nn as nn
from torch_geometric.nn.norm import BatchNorm

def masked_mse_loss(pred, target, mask):
    """
    pred:   (batch_size, 4)
    target: (batch_size, 4)
    mask:   (batch_size, 4) with 1 for present tasks, 0 for missing
    """
    # (pred - target)^2 shape = (batch_size, 4)
    mse = (pred - target) ** 2
    # Multiply by mask to ignore missing tasks
    masked_mse = mse * mask
    # Avoid division by zero
    loss = masked_mse.sum() / (mask.sum() + 1e-8)
    return loss

def masked_mae_loss(pred, target, mask):
    mae = torch.abs(pred - target)
    masked_mae = mae * mask
    return masked_mae.sum() / (mask.sum() + 1e-8)

class GCN(pl.LightningModule):
    def __init__(
        self, 
        num_node_features, 
        hidden_channels, 
        n_tasks=4, 
        dropout=0.2,
        learning_rate=0.01, 
        weight_decay=1e-4, 
        lr_factor=0.1, 
        lr_patience=10
    ):
        super(GCN, self).__init__()
        self.save_hyperparameters()
        
        if not isinstance(hidden_channels, list):
            hidden_channels = [hidden_channels]
        
        self.gconv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        self.gconv_layers.append(GCNConv(num_node_features, hidden_channels[0]))
        self.batch_norms.append(BatchNorm(hidden_channels[0]))
        self.dropouts.append(nn.Dropout(dropout))
        
        # Hidden layers
        for i in range(1, len(hidden_channels)):
            self.gconv_layers.append(GCNConv(hidden_channels[i-1], hidden_channels[i]))
            self.batch_norms.append(BatchNorm(hidden_channels[i]))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Output layer: predict 4 tasks
        self.lin = nn.Linear(hidden_channels[-1], n_tasks)
        
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.n_tasks = n_tasks

    def forward(self, x, edge_index, batch):
        x = x.float()
        edge_index = edge_index.long()
        
        # GCN layers
        for i, gconv in enumerate(self.gconv_layers):
            x = gconv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropouts[i](x)
        
        # Pool
        x = global_mean_pool(x, batch)
        
        # Output shape: (batch_size, 4)
        x = self.lin(x)
        return x

    def training_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index, batch.batch)  # (batch_size, 4)
        y = batch.y  # (batch_size, 4)
        mask = batch.mask  # (batch_size, 4), 1=present, 0=missing

        loss = masked_mse_loss(y_hat, y, mask)
        mae = masked_mae_loss(y_hat, y, mask)

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, batch_size=y.size(0))
        self.log('train_mae', mae, on_step=True, on_epoch=True, logger=True, batch_size=y.size(0))
        
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index, batch.batch)
        y = batch.y
        mask = batch.mask
        
        loss = masked_mse_loss(y_hat, y, mask)
        mae = masked_mae_loss(y_hat, y, mask)

        self.log('val_loss', loss, prog_bar=True, logger=True, batch_size=y.size(0))
        self.log('val_mae', mae, prog_bar=True, logger=True, batch_size=y.size(0))
        
        return loss

    def test_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index, batch.batch)
        y = batch.y
        mask = batch.mask

        loss = masked_mse_loss(y_hat, y, mask)
        mae = masked_mae_loss(y_hat, y, mask)

        self.log('test_loss', loss, prog_bar=True, logger=True, batch_size=y.size(0))
        self.log('test_mae', mae, prog_bar=True, logger=True, batch_size=y.size(0))
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=self.lr_factor, patience=self.lr_patience
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_mae',  
                'frequency': 1,        
                'interval': 'epoch'    
            }
        }
