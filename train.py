#!/usr/bin/env python3

from model import unet
import argparse
import wandb
import os
from wandb.keras import WandbCallback
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
import keras_metrics as km
from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf

seed(1)
set_random_seed(2)

from data import get_training_data

parser = argparse.ArgumentParser(description='Train the witness model.')
parser.add_argument('-e', '--epochs', type=int, help='Number of epochs to run.')
parser.add_argument('-s', '--steps', type=int, help='Number of steps per epoch.')
parser.add_argument('--batch-size', type=int, help='Batch size.')
parser.add_argument('--beta', type=int, help='Degree to prefer positive predictions.')
parser.add_argument('--learning-rate', type=float, help='Learning rate.')
parser.add_argument('--num-predictions', type=int, help='Number of predictions in the validation set to log to W&B.')

augmentation_params = {
    'rotation_range': 0.2,
    'width_shift_range': 0.05,
    'height_shift_range': 0.05,
    'shear_range': 0.05,
    'zoom_range': 0.05,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

def weighted_cross_entropy(beta):
    """
    Weighted cross entry to weight the model to prefer positive predictions.
    """
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        return tf.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)
        return tf.reduce_mean(loss)
    return loss

def main():
    # Get args
    args = parser.parse_args()

    # Init wandb
    run = wandb.init(tensorboard=True)
    run.config.learning_rate = args.learning_rate or 1e-4
    run.config.num_epochs = args.epochs or 100
    run.config.steps_per_epoch = args.steps or 300
    run.config.batch_size = args.batch_size or 8
    run.config.image_size = (288, 512)
    run.config.num_predictions = args.num_predictions or 24
    run.config.beta = args.beta or 50

    wandb.save('*.py')

    training_data_generator = get_training_data(run.config.batch_size,
        'data/train', 'images', 'labels', augmentation_params,
        target_size=run.config.image_size)

    validation_data_generator = get_training_data(run.config.batch_size,
        'data/valid', 'images', 'labels', {},
        target_size=run.config.image_size)

    validation_data_generator_2 = get_training_data(run.config.batch_size,
        'data/valid', 'images', 'labels', {},
        target_size=run.config.image_size)

    os.makedirs('model', exist_ok=True)

    model = unet(image_size=run.config.image_size)
    metrics = ['accuracy', km.precision(), km.recall()]

    model.compile(
        optimizer=Adam(lr=run.config.learning_rate),
        loss=weighted_cross_entropy(run.config.beta),
        # loss='binary_crossentropy',
        metrics=metrics)

    # Save best model
    model_path = 'model/unet_witness.hdf5'
    model_checkpoint = ModelCheckpoint(
        model_path, monitor='loss',
        verbose=1, save_best_only=True)

    # Upload examples to W&B
    wandb_callback = WandbCallback(
        data_type='image',
        predictions=run.config.num_predictions,
        generator=validation_data_generator_2,
        save_model=True,
        monitor='loss',
        mode='min',
        labels=['void', 'puzzle'])

    # Save to tensorboard
    tensorboard_callback = TensorBoard(
        log_dir=wandb.run.dir,
        histogram_freq=0,
        write_graph=True,
        write_images=True)

    callbacks = [model_checkpoint, wandb_callback, tensorboard_callback]

    model.fit_generator(
        training_data_generator,
        validation_data=validation_data_generator,
        validation_steps=run.config.num_predictions,
        steps_per_epoch=run.config.steps_per_epoch,
        epochs=run.config.num_epochs,
        callbacks=callbacks)

    # Upload best model to W&B
    wandb.save(model_path)

if __name__ == '__main__':
    main()
