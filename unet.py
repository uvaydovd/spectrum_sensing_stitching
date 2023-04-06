#Author: Daniel Uvaydov
#Use to train DL network on time series IQs
#Network takes as input 1024 IQs in frequency domain
#Network outputs label for each sub-band or each IQ (google semantic segmentation)

import numpy as np
import argparse
import os
import h5py
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, LeakyReLU, Flatten, Dropout, BatchNormalization, \
    Multiply, GlobalAveragePooling1D, Input, add, SeparableConv1D, Activation, UpSampling1D, Conv2DTranspose, Lambda
from tensorflow.keras.models import Model
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize


# Parse Arguments
parser = argparse.ArgumentParser(description='Training or testing networks with GPU specifications')
parser.add_argument('--dset', type=str, default='./dset.h5', help="filepath of h5 dataset")
parser.add_argument('--normalize', type=bool, default=False, help="choose whether to l2 normalize input or not")
parser.add_argument('--id_gpu', default='0', type=str,
                    help='Choose GPU to use')
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)

tf.compat.v1.disable_eager_execution()


def Conv1DTranspose(input_tensor, filters, kernel_size, strides=1, padding='same'):
    """
        input_tensor: tensor, with the shape (batch_size, time_steps, dims)
        filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
        kernel_size: int, size of the convolution kernel
        strides: int, convolution step size
        padding: 'same' | 'valid'
    """
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

def build_model(batch_size, dim, n_channels, n_classes, shuffle):

    inputs = Input(shape=(dim, n_channels))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = Conv1D(64, 3, strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [128, 256, 512]:
        x = Activation("relu")(x)
        x = SeparableConv1D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = SeparableConv1D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = MaxPooling1D(2, strides=2, padding="same")(x)
        x = Dropout(0.5)(x)

        # Project residual
        residual = Conv1D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [512, 256, 128, 64]:
        x = Activation("relu")(x)
        x = Conv1DTranspose(x, filters, 3, padding="same")
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = Conv1DTranspose(x, filters, 3, padding="same")
        x = BatchNormalization()(x)

        x = UpSampling1D(2)(x)
        x = Dropout(0.5)(x)


        # Project residual
        residual = UpSampling1D(2)(previous_block_activation)
        residual = Conv1D(filters, 1, padding="same")(residual)

        x = add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = Conv1D(n_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = Model(inputs, outputs)
    model.summary()

    return model


def train(model, X, y, batch_size, dim, n_channels, n_classes, shuffle):

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=0),
        tf.keras.callbacks.ModelCheckpoint(
            './tf_model.h5',
            monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=2),
        tf.keras.callbacks.CSVLogger('./tf_model_hist.csv',
                                     separator=',', append=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, verbose=2),
    ]


    model.fit(x=X, y=y, validation_split=0.1, batch_size=batch_size, epochs=250, verbose=1, shuffle=shuffle, callbacks=callbacks)






if __name__ == '__main__':

    #Load data
    #assumes data is h5 file of shape [nsamples,1024,2]
    data_fp = args.dset
    print(data_fp)
    dset = h5py.File(data_fp, 'r')
    X = dset['X'][()]
    y = dset['y'][()]

    #shuffle data
    X, y = shuffle(X, y)
    y = np.expand_dims(y, -1)

    #l2 normalize data
    if args.normalize:
        X_norm = normalize(np.reshape(X,(-1,2)))
        X = np.reshape(X_norm,(-1,1024,2))

    #training params
    params = {'batch_size': 512,
              'dim': X.shape[1],
              'n_channels': 2,
              'n_classes': 6,
              'shuffle': True
              }


    #Build model
    model = build_model(**params)
    adam = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  optimizer=adam,
                  metrics=['accuracy'])

    #train model
    train(model, X, y, **params)

    #convert best checkpoint model to tf model format, will be used later to convert to onnx
    best_model = tf.keras.models.load_model('./tf_model.h5')
    best_model.save('./tf_model')
