#!/usr/bin/env python3
from keras import Model, Input
from keras.layers import Reshape, Conv2D, MaxPooling2D, Dense, Concatenate, \
    Dot, Dropout
from keras.optimizers import Adam
from keras.utils import plot_model

from config import N_SAMPLES, INPUT_FEATURES, OUTPUT_FEATURES


def input_transformation(inputs, K=3):
    r0_0 = Reshape((N_SAMPLES, sum(INPUT_FEATURES), 1))(inputs)
    conv0_0 = Conv2D(16, (1, sum(INPUT_FEATURES)), strides=(1, 1), padding="valid")(r0_0)
    conv0_1 = Conv2D(32, (1, 1), strides=(1, 1), padding="valid")(conv0_0)
    conv0_2 = Conv2D(256, (1, 1), strides=(1, 1), padding="valid")(conv0_1)
    max0_0 = MaxPooling2D(pool_size=(N_SAMPLES, 1), padding="valid")(conv0_2)
    r0_1 = Reshape((-1,))(max0_0)
    d0_0 = Dense(128)(r0_1)
    d0_1 = Dense(64)(d0_0)
    d0_2 = Dense(K * sum(INPUT_FEATURES))(d0_1)
    r0_2 = Reshape([sum(INPUT_FEATURES), K])(d0_2)
    return r0_2


def feature_transformation(inputs, K=3):
    conv2_0 = Conv2D(16, (1, 1), strides=(1, 1), padding="valid")(inputs)
    conv2_1 = Conv2D(32, (1, 1), strides=(1, 1), padding="valid")(conv2_0)
    conv2_2 = Conv2D(256, (1, 1), strides=(1, 1), padding="valid")(conv2_1)
    max2 = MaxPooling2D((N_SAMPLES, 1), padding="valid")(conv2_2)
    r2_0 = Reshape((-1,))(max2)
    d2_0 = Dense(128)(r2_0)
    d2_1 = Dense(64)(d2_0)
    d2_2 = Dense(K * K)(d2_1)
    r2_1 = Reshape([K, K])(d2_2)
    return r2_1


if __name__ == '__main__':
    location = Input(shape=(N_SAMPLES, INPUT_FEATURES[0]), name="Location")
    color = Input(shape=(N_SAMPLES, INPUT_FEATURES[1]), name="Color")
    channel_mix = Concatenate(axis=-1)([location, color])

    tr0 = input_transformation(channel_mix, K=sum(INPUT_FEATURES))
    mul0 = Dot((-1))([channel_mix, tr0])
    r1_0 = Reshape((N_SAMPLES, sum(INPUT_FEATURES), 1))(mul0)
    conv1_0 = Conv2D(16, (1, sum(INPUT_FEATURES)), strides=(1, 1), padding="valid")(r1_0)
    conv1_1 = Conv2D(16, (1, 1), strides=(1, 1), padding="valid")(conv1_0)

    ft0 = feature_transformation(conv1_1, K=16)
    r2 = Reshape((N_SAMPLES, 16))(conv1_1)
    mul1 = Dot((-1))([r2, ft0])
    r1_1 = Reshape((N_SAMPLES, 1, 16))(mul1)

    conv1_2 = Conv2D(16, (1, 1), strides=(1, 1), padding="valid")(r1_1)
    conv1_3 = Conv2D(32, (1, 1), strides=(1, 1), padding="valid")(conv1_2)
    conv1_4 = Conv2D(256, (1, 1), strides=(1, 1), padding="valid")(conv1_3)

    max1 = MaxPooling2D((N_SAMPLES, 1), padding="valid")(conv1_4)
    r3 = Reshape((-1,))(max1)
    d1_0 = Dense(128)(r3)
    drop0 = Dropout(0.3)(d1_0)
    d1_1 = Dense(64)(drop0)
    drop1 = Dropout(0.3)(d1_1)
    d1_2 = Dense(OUTPUT_FEATURES)(drop1)

    model = Model(inputs=[location, color], outputs=[d1_2])

    model.summary()

    # PLOT
    plot_model(model, show_shapes=True)

    # OPTIMIZER
    adam = Adam()
    model.compile(adam, loss="binary_crossentropy", metrics=["acc"])
    model.save("pointnet_raw.hdf5")
