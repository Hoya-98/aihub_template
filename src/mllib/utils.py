# TODO add json to row func
# TODO add jsons to csv func
# TODO retrieve pretrained weights (src in aihub)

from pathlib import Path
import mllib
import re

workspace_path = Path(mllib.config['WORKSPACE_PATH'])

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