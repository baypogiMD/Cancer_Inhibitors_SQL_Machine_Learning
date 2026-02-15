import pandas as pd
from torch_geometric.data import Dataset
from rdkit import Chem
from torch_geometric.data import Data
import torch

class InhibitorDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def len(self):
        return len(self.df)

    def get(self, idx):
        row = self.df.iloc[idx]
        mol = Chem.MolFromSmiles(row.smiles)

        x = torch.tensor([
            [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetHybridization().real,
                atom.GetIsAromatic()
            ]
            for atom in mol.GetAtoms()
        ], dtype=torch.float)

        edge_index = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index += [[i, j], [j, i]]

        return Data(
            x=x,
            edge_index=torch.tensor(edge_index).t(),
            y=torch.tensor([row.pIC50], dtype=torch.float)
        )
