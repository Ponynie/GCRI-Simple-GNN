# data_module.py (Excerpt)

import os
import pandas as pd
import torch
import random
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_smiles
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

class MTL_MoleculeDataset(Dataset):
    def __init__(self, root, data_df: pd.DataFrame, transform=None, pre_transform=None, pre_filter=None):
        self.data_df = data_df
        super(MTL_MoleculeDataset, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(self.data_df.shape[0])]

    def process(self):
        for idx, row in self.data_df.iterrows():
            smiles = row['smiles']
            data = from_smiles(smiles)

            # 4 tasks: NP_KV, P_KV, NP_VAN, P_VAN
            # Convert NaN to 0.0 for safe tensor creation
            np_kv_val = row['NP_KV_value'] if not pd.isna(row['NP_KV_value']) else 0.0
            p_kv_val  = row['P_KV_value']  if not pd.isna(row['P_KV_value'])  else 0.0
            np_van_val= row['NP_VAN_value']if not pd.isna(row['NP_VAN_value']) else 0.0
            p_van_val = row['P_VAN_value'] if not pd.isna(row['P_VAN_value'])  else 0.0

            # y is shape [4]
            data.y = torch.tensor([ 
                np_kv_val,
                p_kv_val,
                np_van_val,
                p_van_val
            ], dtype=torch.float)

            # Create a mask vector (1 = present, 0 = missing)
            data.mask = torch.tensor([
                row['NP_KV'], 
                row['P_KV'], 
                row['NP_VAN'], 
                row['P_VAN']
            ], dtype=torch.float)

            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data
    
# data_module.py (Excerpt)
class MTL_MoleculeDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, batch_size=32, num_workers=4, test_size=0.2, random_state=42):
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.random_state = random_state
        self.data_df = None

    def setup(self, stage=None):
        if self.data_df is None:
            self.data_df = pd.read_csv(self.csv_path)
            
            # Optional: if you have 'stratify_group', use it in train_test_split
            self.train_df, test_val_df = train_test_split(
                self.data_df,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=self.data_df['stratify_group'] if 'stratify_group' in self.data_df.columns else None
            )
            
            self.test_df, self.val_df = train_test_split(
                test_val_df,
                test_size=0.3,
                random_state=self.random_state,
                stratify=test_val_df['stratify_group'] if 'stratify_group' in test_val_df.columns else None
            )

        if stage == 'fit' or stage is None:
            self.train_dataset = MTL_MoleculeDataset(
                root=os.path.join('data/ALL-KVVAN-ISORAMP', 'train'),
                data_df=self.train_df.reset_index(drop=True)
            )
            self.val_dataset = MTL_MoleculeDataset(
                root=os.path.join('data/ALL-KVVAN-ISORAMP', 'val'),
                data_df=self.val_df.reset_index(drop=True)
            )

        if stage == 'test' or stage is None:
            self.test_dataset = MTL_MoleculeDataset(
                root=os.path.join('data/ALL-KVVAN-ISORAMP', 'test'),
                data_df=self.test_df.reset_index(drop=True)
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
