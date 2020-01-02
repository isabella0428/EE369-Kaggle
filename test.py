"""
Testing (Training, Testing, Valid)

Author: Yi Lyu
Email: 
Date: 
Part of code adopted from duducheng@github 
(Give credit for him!)
"""
from mylib.dataloader.dataset import get_train_loader, TestDataset, get_test_loader, ValDataset
from mylib.models import densesharp, metrics, losses, densenet

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.models import load_model
import numpy as np
import pandas as pd


def configure():
    '''Configure the platform'''
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    num_cores = 4
    num_CPU = 32
    num_GPU = 1
    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
    session = tf.Session(config=config)
    KTF.set_session(session)
   
    
def main(crop_size=[8, 8, 8], random_move=None, weight_decay=0.0, lr=0.01):
    '''Main function to predict the score'''
    test_dataset = TestDataset(crop_size=crop_size,move=random_move)
    test_loader = get_test_loader(test_dataset, batch_size=1)
    model = densenet.get_compiled(weights="weight.h5", output_size=1,
                                    optimizer=Adam(lr=lr))
    result = model.predict_generator(generator=test_loader, steps=117, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
    np.save("result/result.npy", result)


def printResult():
    '''Print result.csv'''
    result = np.load("result/result.npy")
    csv = pd.read_csv("data/test/test.csv")
    for i in range(csv.shape[0]):
        csv.iloc[i, 1] = result[i, 0]
    csv.columns = ['Id', 'Predicted']
    # print(csv)
    csv.to_csv("submission.csv" , index=None)
    
if __name__ == '__main__':
    configure()
    main()
    printResult()
