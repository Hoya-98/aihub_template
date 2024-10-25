from pathlib import Path

import mllib
import mllib.config as config
from mllib.dataset import *

if __name__ == '__main__':
    meta_path = Path(config.workspace.out_folder) / 'meta.csv'

    train_dataset = AIHubBBoxBurnDatset(meta_path=meta_path, mode="Training")
    val_dataset   = AIHubBBoxBurnDatset(meta_path=meta_path, mode="Validation")
    
    
