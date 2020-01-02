"""
Dataset (Training, Testing, Valid)

Author: Yi Lyu
Email: 
Date: 
Part of code adopted from duducheng@github 
(Give credit for him!)
"""
from mylib.dataloader.dataset import get_test_loader, TrainDataset, get_train_loader, ValDataset
from mylib.models import densesharp, metrics, losses, densenet

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


def configure():
    '''Configure the platform'''
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    num_cores = 4
    num_CPU = 32
    num_GPU = 1
    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
    session = tf.Session(config=config)
    KTF.set_session(session)
   
 
def main(batch_size=20, crop_size=[8, 8, 8], random_move=None, learning_rate=1.e-4,
         weight_decay=0, save_folder='save', epochs=100):
    '''Training main function'''
    # Get training dataset
    train_dataset = TrainDataset(crop_size=crop_size, move=None)
    train_loader = get_train_loader(train_dataset, batch_size=batch_size)

    # Get valid dataset
    val_dataset = ValDataset(crop_size=crop_size, move=None)
    val_loader = get_train_loader(ValDataset, batch_size=batch_size)
    
    model = densenet.get_compiled(output_size=1,
                                    optimizer=Adam(lr=learning_rate),
                                    weight_decay=weight_decay)

    # Callback function
    checkpointer = ModelCheckpoint(filepath='tmp/%s/weights.{epoch:02d}.h5' % save_folder, verbose=1,
                                   period=1, save_weights_only=True)
    csv_logger = CSVLogger('tmp/%s/training.csv' % save_folder)
    tensorboard = TensorBoard(log_dir='tmp/%s/logs/' % save_folder)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.334, patience=10,
                                   verbose=1, mode='min', epsilon=1.e-5, cooldown=2, min_lr=0)
    
    model.fit_generator(generator=train_loader, steps_per_epoch=len(train_dataset), max_queue_size=500, workers=1,
                        epochs=epochs, validation_data=val_loader, validation_steps=len(val_dataset),
                        callbacks=[checkpointer, csv_logger, tensorboard, lr_reducer])


if __name__ == '__main__':
   configure()
   main()
