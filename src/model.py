import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import pytorch_lightning as pl
import torch.nn as nn
from torch_geometric.nn.norm import BatchNorm

class GCN(pl.LightningModule):
    def __init__(self, num_node_features, hidden_channels, n_tasks = 1, dropout=0.2, learning_rate=0.01, weight_decay=1e-4, lr_factor=0.1, lr_patience=10):
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
        
        # Output layer
        self.lin = torch.nn.Linear(hidden_channels[-1], n_tasks)
        
        self.criterion = nn.MSELoss()
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience  

    def forward(self, x, edge_index, batch):
        # Ensure x is float
        x = x.float()
        
        # Ensure edge_index is long
        edge_index = edge_index.long()
        
        # Obtain node embeddings
        for i, gconv in enumerate(self.gconv_layers):
            x = gconv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropouts[i](x)
        
        # Use global_mean_pool to aggregate node features for each graph in the batch
        x = global_mean_pool(x, batch)
        
        x = self.lin(x)
        return x.squeeze(-1)  # Remove the last dimension to match target shape

    def training_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index, batch.batch)
        y = batch.y
        
        loss = self.criterion(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, batch_size=y.size(0))
        self.log('train_mae', mae, on_step=True, on_epoch=True, logger=True, batch_size=y.size(0))
        
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index, batch.batch)
        y = batch.y
        
        loss = self.criterion(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        
        self.log('val_loss', loss, prog_bar=True, logger=True, batch_size=y.size(0))
        self.log('val_mae', mae, prog_bar=True, logger=True, batch_size=y.size(0))
        
        return loss

    def test_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index, batch.batch)
        y = batch.y
        
        loss = self.criterion(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        
        self.log('test_loss', loss, prog_bar=True, logger=True, batch_size=y.size(0))
        self.log('test_mae', mae, prog_bar=True, logger=True, batch_size=y.size(0))
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.lr_factor, patience=self.lr_patience)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_mae',  
                'frequency': 1,        
                'interval': 'epoch'    
            }
        }