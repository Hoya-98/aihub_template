from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import mllib
from pathlib import Path

class BurnDataset(Dataset):
    def __init__(self, mode: str, shape: str):
        assert mode in ['Sample', 'Training', 'Validation']
        assert shape in ['Single', 'Time series']

        workspace_path = Path(mllib.config['WORKSPACE_PATH'])

        self.mode = mode
        self.shape = shape
        self.df = None
    
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
