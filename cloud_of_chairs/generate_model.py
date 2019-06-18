#!/usr/bin/env python3
from keras import Model, Input
from keras.layers import Dense, Softmax, Conv3D
from keras.optimizers import SGD
from keras.utils import plot_model

from config import INPUT_SHAPE, RAW_MODEL, OUTPUT_FEATURES

if __name__ == '__main__':
    grid = Input(shape=INPUT_SHAPE)
    conv0 = Conv3D(8, (3, 3, 3), strides=(1, 1, 1), padding="same")(grid)
    conv3 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding="same")(conv0)

    d0 = Dense(32)(conv3)
    d1 = Dense(8)(d0)
    d2 = Dense(OUTPUT_FEATURES)(d1)
    soft = Softmax(axis=-1)(d2)

    model = Model(inputs=[grid], outputs=[soft])

    model.summary()

    # PLOT
    plot_model(model, show_shapes=True)

    # OPTIMIZER
    sgd = SGD(lr=0.00001, momentum=0.9, decay=0.0005)
    model.compile(sgd, loss="binary_crossentropy", metrics=["acc"])
    model.save(RAW_MODEL)
