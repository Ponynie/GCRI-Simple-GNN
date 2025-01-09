# data_module.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_smiles
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import random

random.seed(42)

class MoleculeDataset(Dataset):
    def __init__(self, root, data_df: pd.DataFrame, transform=None, pre_transform=None, pre_filter=None, ri2d=False):
        self.data_df = data_df
        self.ri2d = ri2d
        super(MoleculeDataset, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(self.data_df.shape[0])]

    def process(self):
        for idx, row in self.data_df.iterrows():
            smiles = row['smiles']
            data = from_smiles(smiles)
            
            if self.ri2d:
                ri = row['ri']
                ri_2 = row['ri_2']
                data.y = torch.tensor([ri, ri_2], dtype=torch.float).reshape(1, 2)
            else:
                ri = row['ri']
                data.y = torch.tensor([ri], dtype=torch.float)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

class MoleculeDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, batch_size=32, num_workers=4, test_size=0.2, random_state=42, ri2d=False):
        super().__init__()
        self.random_state = random_state
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.data_dir = os.path.dirname(csv_path)
        self.data_df = None
        self.ri2d = ri2d
        
    def setup(self, stage=None):
        if self.data_df is None:
            # Read CSV file
            self.data_df = pd.read_csv(self.csv_path)

            # First split: separate train data from test+val data
            self.train_df, test_val_df = train_test_split(
                self.data_df, 
                test_size=self.test_size, 
                random_state=self.random_state, 
                stratify=self.data_df['stratify_group']
            )
            
            # Second split: separate test and validation data
            self.test_df, self.val_df = train_test_split(
                test_val_df, 
                test_size=0.3,  # 30% of test_val_data will be validation
                random_state=self.random_state, 
                stratify=test_val_df['stratify_group']
            )

        # Create datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = MoleculeDataset(
                root=os.path.join(self.data_dir, 'train'),
                data_df=self.train_df.reset_index(drop=True),
                ri2d=self.ri2d
            )
            self.val_dataset = MoleculeDataset(
                root=os.path.join(self.data_dir, 'val'),
                data_df=self.val_df.reset_index(drop=True),
                ri2d=self.ri2d
            )
        if stage == 'test' or stage is None:
            self.test_dataset = MoleculeDataset(
                root=os.path.join(self.data_dir, 'test'),
                data_df=self.test_df.reset_index(drop=True),
                ri2d=self.ri2d
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
class MoleculeDataModuleSplitted(pl.LightningDataModule):
    def __init__(self, train_path, val_path, test_path, batch_size=32, num_workers=4, random_state=42, ri2d=False):
        super().__init__()
        self.random_state = random_state
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = os.path.dirname(train_path)
        self.ri2d = ri2d
        
    def setup(self, stage=None):
        # Create datasets
        if stage == 'fit' or stage is None:
            train_df = pd.read_csv(self.train_path)
            val_df = pd.read_csv(self.val_path)
            self.train_dataset = MoleculeDataset(
                root=os.path.join(self.data_dir, 'train'),
                data_df=train_df,
                ri2d=self.ri2d
            )
            self.val_dataset = MoleculeDataset(
                root=os.path.join(self.data_dir, 'val'),
                data_df=val_df,
                ri2d=self.ri2d
            )
        if stage == 'test' or stage is None:
            test_df = pd.read_csv(self.test_path)
            self.test_dataset = MoleculeDataset(
                root=os.path.join(self.data_dir, 'test'),
                data_df=test_df,
                ri2d=self.ri2d
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)