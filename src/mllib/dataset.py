from functools import cached_property
from pathlib import Path
from typing import List

import pandas as pd
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from torch.utils.data import Dataset

import mllib
import mllib.config as config
from mllib.config import MODES, SHAPES
from mllib.utils import generate_meta


train_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225])
])

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225])
])

class AIHubBurnDataset(Dataset):
    def __init__(self,
                 meta_path: str,
                 transform: T = None,
                 mode: List[str] = MODES,
                 shape: List[str] = SHAPES,
                 *args, **kwags):
        super().__init__(*args, **kwags)

        meta_path = Path(meta_path)
        if not meta_path.exists():
            generate_meta(workspace_path=config.workspace.path,
                          out_path=f"{config.workspace.out_folder}/meta.csv")

        if not isinstance(mode, list):
            mode = [mode]
        if not isinstance(shape, list):
            shape = [shape]

        self.mode = mode
        self.shape = shape
        self.transform = transform
        self._df = pd.read_csv(meta_path)
    
    @cached_property
    def df(self):
        return self._df[
            (self._df['path.mode'].isin(self.mode))
            & (self._df['path.shape'].isin(self.shape))
        ]
    
    def __add__(self, other):
        self.transform = None
        self.df = pd.concat([self.df, other.df], axis=0).drop_duplicates()
        self.df = self.df.reset_index(drop=True)
    
    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        label = row['patient.stage']
        img = Image.open(row['path.image']).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label


class AIHubBBoxBurnDatset(AIHubBurnDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bbox = row['annotations.bbox']
        x, y, dx, dy = bbox = list(map(int, bbox.split(',')))

        label = row['patient.stage']
        img = Image.open(row['path.image']).convert('RGB')
        img = ImageOps.exif_transpose(img)
        img = img.crop((x, y, x+dx, y+dy))

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label
