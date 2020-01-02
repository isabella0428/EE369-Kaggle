"""
Dataset (Training, Testing, Valid)

Author: Yi Lyu
Email: 
Date: 
Part of code adopted from duducheng@github 
(Give credit for him!)
"""
from collections.abc import Sequence
import random
import os

import numpy as np
from mylib.dataloader.path_manager import TrainPathManager, ValPathManager, TestPathManager
from mylib.utils.misc import rotation, reflection, crop, random_center, _triple


class TestDataset(Sequence):
    '''Testing dataset'''

    def __init__(self, crop_size=32, move=None):
        """
        Init Testing dataset
        
        params
              crop_size: the crop size
                   move: the random move   
        """
        PATH = TestPathManager()
        self.info = PATH.info
        self.path = PATH.nodule_path
        index = list(self.info.index)
        self.index = tuple(sorted(index))
        self.transform = Transform(crop_size, move)

    def __getitem__(self, item):
        '''Get next item'''
        name = self.info.loc[self.index[item], 'name']
        with np.load(os.path.join(self.path, '%s.npz' % name)) as npz:
            print(name)
            voxel = self.transform(npz['voxel'])
        return voxel

    @staticmethod
    def _collate_fn(data):
        '''Get next batch'''
        xs = []
        ys = []
        for x in data:
            xs.append(x)
        return np.array(xs)

    
    def __len__(self):
        return len(self.index)
    
    
class TrainDataset(Sequence):
    '''Training dataset.'''

    def __init__(self, crop_size=32, move=None):
        """
        Init training dataset
        
        params
              crop_size: the crop size
                   move: the random move   
        """
        PATH = TrainPathManager()
        self.info = PATH.info
        self.path = PATH.nodule_path
        index = list(self.info.index)
        self.label = np.array(self.info['label'])
        self.index = tuple(sorted(index))
        # Apply transformation
        self.transform = Transform(crop_size, move)

    def __getitem__(self, item):
        '''Get next item'''
        name = self.info.loc[self.index[item], 'name']
        with np.load(os.path.join(self.path, '%s.npz' % name)) as npz:
            voxel = self.transform(npz['voxel'])
        return voxel, self.label[item]

    @staticmethod
    def _collate_fn(data):
        '''Get next batch'''
        xs = []
        ys = []
        for x, y in data:
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    
    def __len__(self):
        '''Return the length of dataset'''
        return len(self.index)

class ValDataset(TrainDataset):
    '''Valid dataset.'''

    def __init__(self, crop_size=32, move=None):
        """
        Init training dataset
        
        params
              crop_size: the crop size
                   move: the random move   
        """
        PATH = ValPathManager()
        self.info = PATH.info
        self.path = PATH.nodule_path
        index = list(self.info.index)
        self.label = np.array(self.info['label'])
        self.index = tuple(sorted(index))
        # Apply transformation
        self.transform = Transform(crop_size, move)

    
    
def get_train_loader(dataset, batch_size):
    """Get training/valid dataset loader"""
    total_size = len(dataset)
    index_generator = shuffle_iterator(range(total_size))
    while True:
        data = []
        for _ in range(batch_size):
            idx = next(index_generator)
            data.append(dataset[idx])
        yield dataset._collate_fn(data)

    
def get_test_loader(dataset, batch_size):
    """Get testing dataset loader"""
    total_size = len(dataset)
    index_generator = normal_iterator(range(total_size))
    while True:
        data = []
        for _ in range(batch_size):
            idx = next(index_generator)
            data.append(dataset[idx])
        yield dataset._collate_fn(data)
        

class Transform:
    """
    The online data augmentation, including:
        1) random move the center by `move`
        2) rotation 90 degrees increments
        3) reflection in any axis
    """
    def __init__(self, size, move):
        self.size = _triple(size)
        self.move = move

    def __call__(self, arr, aux=None):
        shape = arr.shape
        if self.move is not None:
            center = random_center(shape, self.move)
            arr_ret = crop(arr, center, self.size)
            angle = np.random.randint(4, size=3)
            arr_ret = rotation(arr_ret, angle=angle)
            axis = np.random.randint(4) - 1
            arr_ret = reflection(arr_ret, axis=axis)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = rotation(aux_ret, angle=angle)
                aux_ret = reflection(aux_ret, axis=axis)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret
        else:
            center = np.array(shape) // 2
            arr_ret = crop(arr, center, self.size)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret


def shuffle_iterator(iterator):
    '''Return iterator in a round fashion(random shuffle)'''
    index = list(iterator)
    total_size = len(index)
    i = 0
    random.shuffle(index)
    while True:
        yield index[i]
        i += 1
        if i >= total_size:
            i = 0
            random.shuffle(index)

def normal_iterator(iterator):
    '''Return iterator in a round fashion(no random shuffle)'''
    # Don't shuffle index
    index = list(iterator)
    total_size = len(index)
    i = 0
    while True:
        yield index[i]
        i += 1
        if i >= total_size:
            i = 0
