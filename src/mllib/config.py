import os
from pprint import pprint

config = {
    'dev': {
        'WORKSPACE_ENV': 'dev',
        'WORKSPACE_PATH': '../../workspace',
    },
    'prod': {
        'WORKSPACE_ENV': 'prod',
        'WORKSPACE_PATH': '/workspace',
    }
}

env = os.getenv('WORKSPACE_ENV', 'dev')

if env == 'dev':
    config = config['dev']
elif env == 'prod':
    config = config['prod']
else:
    raise NotImplementedError('specify WORKSPACE_ENV between [dev, prod].')