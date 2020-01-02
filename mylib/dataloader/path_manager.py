"""
PathManager (Training, Testing, Valid)

Author: Yi Lyu
Email: 
Date: 
Part of code adopted from duducheng@github 
(Give credit for him!)
"""
import os
import json
import numpy as np

class PathManager:
    '''Base class for path manager'''
    def __init__(self, cfg_path=None):
        self.environ = parse_environ(cfg_path)

    @property
    def base(self):
        '''The base address of the data'''
        return self.environ['DATASET']

    @property
    def info(self):
        '''Return csv info'''
        import pandas as pd
        df = pd.read_csv(os.path.join(self.base, 'base.csv'))
        return df

    @property
    def nodule_path(self):
        '''Return nodule data'''
        return self.base
    
def parse_environ(cfg_path=None):
    '''Parse environment variables'''
    if cfg_path is None:
        cfg_path = os.path.join(os.path.dirname(__file__), "ENVIRON")
    assert os.path.exists(cfg_path), "`ENVIRON` does not exists."
    with open(cfg_path) as f:
        environ = json.load(f)
    return environ

class TrainPathManager(PathManager):
    '''Training data path manager'''
    @property
    def base(self):
        return self.environ['TRAIN_DATASET']
    
    @property
    def info(self):
        '''Return csv info'''
        import pandas as pd
        df = pd.read_csv(os.path.join(self.base, 'train.csv'))
        return df

class ValPathManager(PathManager):
    '''Valid data path manager'''
    @property
    def base(self):
        return self.environ['VAL_DATASET']
    
    @property
    def info(self):
        '''Return csv info'''
        import pandas as pd
        df = pd.read_csv(os.path.join(self.base, 'val.csv'))
        return df


class TestPathManager(PathManager):
    '''Test data path manager'''
    @property
    def base(self):
        return self.environ['TEST_DATASET']
    
    @property
    def info(self):
        '''Return csv info'''
        import pandas as pd
        df = pd.read_csv(os.path.join(self.base, 'test.csv'))
        return df



