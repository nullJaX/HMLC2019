#!/usr/bin/env python3
from keras import Model, Input
from keras.layers import Reshape, Dense, Concatenate, \
    Dot, Dropout, Softmax, Conv1D, BatchNormalization, Permute, ReLU, \
    MaxPooling1D
from keras.optimizers import SGD
from keras.utils import plot_model

from config import N_SAMPLES, INPUT_FEATURES, OUTPUT_FEATURES, RAW_MODEL


def transformation(inputs, output_dim):
    conv0 = ReLU()(BatchNormalization()(Conv1D(16, (1,))(inputs)))
    conv1 = ReLU()(BatchNormalization()(Conv1D(int(N_SAMPLES/8), (1,))(conv0)))
    conv2 = ReLU()(BatchNormalization()(Conv1D(N_SAMPLES, (1,))(conv1)))
    max0 = MaxPooling1D(pool_size=N_SAMPLES)(conv2)
    r0 = Reshape((-1, N_SAMPLES))(max0)
    d0 = ReLU()(BatchNormalization()(Dense(int(N_SAMPLES/2))(r0)))
    d1 = ReLU()(BatchNormalization()(Dense(int(N_SAMPLES/4))(d0)))
    d2 = Dense(output_dim*output_dim)(d1)
    r1 = Permute((2, 1))(Reshape((output_dim, output_dim))(d2))
    return r1


if __name__ == '__main__':
    location = Input(shape=(N_SAMPLES, INPUT_FEATURES[0]), name="Location")
    color = Input(shape=(N_SAMPLES, INPUT_FEATURES[1]), name="Color")
    features = Concatenate(axis=-1)([location, color])

    input_transform = transformation(features, sum(INPUT_FEATURES))
    dot = Dot(axes=-1)([features, input_transform])
    conv3 = ReLU()(BatchNormalization()(Conv1D(64, (1,))(dot)))
    conv4 = ReLU()(BatchNormalization()(Conv1D(64, (1,))(conv3)))
    features_transform = transformation(conv4, 64)
    dot1 = Dot(axes=-1)([conv4, features_transform])
    # trans3 = Permute((2, 1))(dot1)
    conv5 = ReLU()(BatchNormalization()(Conv1D(64, (1,))(dot1)))
    conv6 = ReLU()(BatchNormalization()(Conv1D(128, (1,))(conv5)))
    conv7 = ReLU()(BatchNormalization()(Conv1D(N_SAMPLES, (1,))(conv6)))
    max0 = MaxPooling1D(N_SAMPLES)(conv7)
    r0 = Reshape((N_SAMPLES, -1))(max0)

    d1 = ReLU()(BatchNormalization()(Dense(int(N_SAMPLES/2))(r0)))
    d2 = ReLU()(BatchNormalization()(Dense(int(N_SAMPLES/4))(d1)))
    drop = Dropout(0.3)(d2)
    d3 = Dense(OUTPUT_FEATURES)(drop)
    soft = Softmax(axis=1)(d3)

    model = Model(inputs=[location, color], outputs=[soft])

    model.summary()

    # PLOT
    plot_model(model, show_shapes=True)

    # OPTIMIZER
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.005)
    model.compile(sgd, loss="binary_crossentropy", metrics=["acc"])
    model.save(RAW_MODEL)
