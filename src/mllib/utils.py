import configparser
import json
import os
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from mllib.config import CLASS_MAP, MODES, SHAPES


class dotdict(dict):
    def __getattr__(self, k):
        v = self.get(k)

        if isinstance(v, dict):
            return dotdict(v)

        elif isinstance(v, configparser.ConfigParser):
            return dotdict({section: dict(v[section]) for section in v.sections()})

        elif isinstance(v, configparser.SectionProxy):
            return dotdict({option: v[option] for option in v})

        return v

    def __setattr__(self, k, v):
        self[k] = v

    def __delattr__(self, k):
        del self[k]
    
    def __repr__(self):
        return f"dotdict({super().__repr__()}"


def get_config(file_path=None):
    config = configparser.ConfigParser()
    if file_path:
        config.read(file_path)
        return dotdict(config)

    ENVENVIRONMENT = os.getenv('ENVENVIRONMENT', 'dev')
    assert ENVENVIRONMENT in ['dev', 'prod']
    
    config.read(f"config.{ENVENVIRONMENT}.ini")
    return dotdict(config)


def flatten_dict(d, parent_key='', sep='.', exclude_regex=None):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if exclude_regex is not None and re.search(exclude_regex, new_key):
            continue

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep, exclude_regex=exclude_regex).items())
        else:
            if isinstance(v, np.ndarray):
                v = v.tolist()
            items.append((new_key, v))
    
    return dict(items)


def dataset_path(mode: str,
                 shape: str,
                 workspace_path: str):
    assert mode in MODES
    assert shape in SHAPES

    return Path(workspace_path) / 'dataset' / '084.화상 이미지 및 임상 데이터' / '01-1.정식개방데이터' / mode / '01.원천데이터' / shape, \
           Path(workspace_path) / 'dataset' / '084.화상 이미지 및 임상 데이터' / '01-1.정식개방데이터' / mode / '02.라벨링데이터' / shape


def generate_meta(workspace_path: str,
                  out_path: str,
                  modes: List[str] = MODES,
                  shapes: List[str] = SHAPES):

    modes = modes if isinstance(modes, list) else [modes]
    shapes = shapes if isinstance(shapes, list) else [shapes]

    paths = []
    for mode in modes:
        for shape in shapes:
            image_folder, label_folder = dataset_path(mode, shape, workspace_path)
            for class_name in CLASS_MAP.keys():
                paths.extend([[mode, shape, class_name, image_folder, label_folder, p.stem] for p in label_folder.glob('**/*.json')])

    data = []
    failed = []
    for mode, shape, class_name, image_folder, label_folder, stem in tqdm(paths):
        image_path = image_folder / class_name / f"{stem}.jpg"
        label_path = label_folder / class_name / f"{stem}.json"
        if image_path.exists() and label_path.exists():
            try:
                with open(label_path, 'r') as f:
                    d = json.load(f)
                    for i, (k, v) in enumerate(d['annotations'].items()):
                        _d = d.copy()
                        _d['path'] = {
                            'image': str(image_path),
                            'label': str(label_path),
                            'mode': mode,
                            'shape': shape,
                            'classname': class_name,
                        }

                        def list2csv(l):
                            return ','.join(map(str, l))

                        _d['annotations'] = {
                            'id': i,
                            'label': k ,
                            'bbox': list2csv(v['bbox'][i]),
                            'segmentation': list2csv(v['segmentation'][i]),
                        }
                        data.append(flatten_dict(_d))

            except:
                failed.append(stem)
                pass
    
    df = pd.DataFrame(data)
    df.to_csv(out_path, index=False)

    print('done!')