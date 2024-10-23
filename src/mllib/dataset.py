from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class BurnDataset(Dataset):
    def __init__(self, mode: str, shape: str):
        self.mode = mode
        self.shape = shape
        self.df = None
    
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
