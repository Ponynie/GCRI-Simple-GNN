# model.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import pytorch_lightning as pl
import torch.nn as nn
from torch_geometric.nn.norm import BatchNorm

def masked_weighted_mse_loss(pred, target, mask, weights):
    mse = (pred - target) ** 2
    weighted_mse = mse * mask * weights
    sum_mse = weighted_mse.sum()
    sum_mask_weight = (mask * weights).sum()
    loss = sum_mse / (sum_mask_weight + 1e-8)
    return loss

def masked_weighted_mae_loss(pred, target, mask, weights): # MAE
    mae = torch.abs(pred - target) 
    weighted_mae = mae * mask * weights
    sum_mae = weighted_mae.sum()
    sum_mask_weight = (mask * weights).sum()
    loss = sum_mae / (sum_mask_weight + 1e-8)
    return loss
 
def masked_mae_loss(pred, target, mask):
    mae = torch.abs(pred - target)  # shape: (batch_size, n_tasks)
    masked_mae = mae * mask  # Only consider valid tasks
    sum_mae_per_task = masked_mae.sum(dim=0)  # Sum over samples for each task
    sum_mask_per_task = mask.sum(dim=0)  # Count of valid samples per task

    # Calculate MAE per task (unweighted)
    per_task_mae = sum_mae_per_task / (sum_mask_per_task + 1e-8)  # shape: (n_tasks,)
    avg_mae = per_task_mae.mean()  # Average MAE across tasks
    
    return avg_mae, per_task_mae

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
        lr_patience=10,
        task_weights=None 
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
        
        # Output layer for 4 tasks
        self.lin = nn.Linear(hidden_channels[-1], n_tasks)
        
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.n_tasks = n_tasks

        # Register the task_weights as a buffer so it moves with the model
        if task_weights is not None:
            # Convert to tensor
            self.register_buffer("task_weights", torch.tensor(task_weights, dtype=torch.float))
        else:
            # Default to all ones if none provided
            self.register_buffer("task_weights", torch.ones(n_tasks, dtype=torch.float))

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
        y_hat = self(batch.x, batch.edge_index, batch.batch)  # Predicted values
        y = batch.y # Ground truth values
        mask = batch.mask  # shape (batch_size, 4)

        # MSE loss (weighted, used for training)
        mse_loss = masked_weighted_mse_loss(y_hat, y, mask, self.task_weights)

        # Unweighted MAE (for logging)
        avg_mae_unweighted, per_task_mae_unweighted = masked_mae_loss(y_hat, y, mask)

        # Log overall loss
        self.log('train_loss', mse_loss, on_step=True, on_epoch=True, logger=True, batch_size=y.size(0))
        self.log('train_mae_unweight_avg', avg_mae_unweighted, on_step=True, on_epoch=True, logger=True, batch_size=y.size(0))
        
        # Log per-task unweighted MAE
        for i in range(self.n_tasks):
            self.log(f'train_mae_unweight_task_{i}', per_task_mae_unweighted[i], on_step=True, on_epoch=True, logger=True, batch_size=y.size(0))
        
        return mse_loss  # Loss remains MSE for backpropagation

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index, batch.batch)  # Predictions
        y = batch.y  # Ground truth values
        mask = batch.mask  # Task-specific mask

        # MSE Loss for validation
        mse_loss = masked_weighted_mse_loss(y_hat, y, mask, self.task_weights)

        # Unweighted MAE (for logging)
        avg_mae_unweighted, per_task_mae_unweighted = masked_mae_loss(y_hat, y, mask)

        # Log overall metrics
        self.log('val_loss', mse_loss, prog_bar=True, logger=True, batch_size=y.size(0))
        self.log('val_mae_unweight_avg', avg_mae_unweighted, prog_bar=True, logger=True, batch_size=y.size(0))
        
        # Log unweighted per-task MAE
        for i in range(self.n_tasks):
            self.log(f'val_mae_unweight_task_{i}', per_task_mae_unweighted[i], prog_bar=True, logger=True, batch_size=y.size(0))
        
        return mse_loss  # Return MSE loss for validation

    def test_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index, batch.batch)  # Predictions
        y = batch.y  # Ground truth values
        mask = batch.mask  # Task-specific mask

        # MSE Loss for testing
        mse_loss = masked_weighted_mse_loss(y_hat, y, mask, self.task_weights)

        # Unweighted MAE (for logging)
        avg_mae_unweighted, per_task_mae_unweighted = masked_mae_loss(y_hat, y, mask)

        # Log overall metrics
        self.log('test_loss', mse_loss, prog_bar=True, logger=True, batch_size=y.size(0))
        self.log('test_mae_unweight_avg', avg_mae_unweighted, prog_bar=True, logger=True, batch_size=y.size(0))
        
        # Log unweighted per-task MAE
        for i in range(self.n_tasks):
            self.log(f'test_mae_unweight_task_{i}', per_task_mae_unweighted[i], prog_bar=True, logger=True, batch_size=y.size(0))
        
        return mse_loss  # Return MSE loss for testing
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=self.lr_factor, patience=self.lr_patience
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  
                'frequency': 1,        
                'interval': 'epoch'    
            }
        }
