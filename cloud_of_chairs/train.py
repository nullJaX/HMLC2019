#!/usr/bin/env python3
import random

import numpy
from keras.callbacks import TerminateOnNaN, ModelCheckpoint
from keras.engine.saving import load_model

from config import RAW_MODEL, OUTPUT_MODEL, BATCH_SIZE, EPOCHS, \
    VALIDATION_SPLIT, ITEMS, N_SAMPLES


def init_callbacks():
    terminator = TerminateOnNaN()
    checkpointer = ModelCheckpoint(
        "./out/checkpoints/model_{epoch:02d}_{loss:.2f}_{val_loss:.2f}_{acc:.2f}_{val_acc:.2f}.hdf5",
        monitor="val_loss", save_weights_only=False, mode="min", period=5)
    return [terminator, checkpointer]


def load_data():
    data = []
    for i in range(1, 6):
        X_file, Y_file = [], []
        with open("./testS/testS{0!s}.in".format(i)) as fin, open(
                "./testS/testS{0!s}.out".format(i)) as fout:
            for line in fin:
                x, y, z, r, g, b = map(float, line.split())
                r /= 255
                g /= 255
                b /= 255
                X_file.append([x, y, z, r, g, b])
            for line in fout:
                Y_file.append([float(line)])
        X_file = numpy.array(X_file)
        Y_file = numpy.array(Y_file)
        data_file = numpy.concatenate((X_file, Y_file), axis=-1)
        data.append(data_file)
    return data


def generator(data):
    while True:
        batch_location = []
        batch_color = []
        batch_output = []
        for i in range(BATCH_SIZE):
            room = i % 5
            r = numpy.random.randint(data[room].shape[0], size=N_SAMPLES)
            d = data[room][r, :]
            x_location = d[:, 0:3].tolist()
            x_color = d[:, 3:6].tolist()
            y = d[:, 6].tolist()

            batch_location.append(x_location)
            batch_color.append(x_color)
            batch_output.append(y)
        batch_location = numpy.array(batch_location)
        batch_color = numpy.array(batch_color)
        batch_output = numpy.array(batch_output)
        batch_output = numpy.expand_dims(batch_output, axis=-1)
        yield ([batch_color], [batch_output])


if __name__ == '__main__':
    callbacks = init_callbacks()
    data = load_data()
    model = load_model(RAW_MODEL)
    # model = create_model()
    model.fit_generator(generator(data),
                        steps_per_epoch=int(numpy.ceil(
                            ((1 - VALIDATION_SPLIT) * ITEMS) / BATCH_SIZE)),
                        epochs=EPOCHS,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=generator(data),
                        validation_steps=int(numpy.ceil(
                            (VALIDATION_SPLIT * ITEMS) / BATCH_SIZE)),
                        max_queue_size=10,
                        shuffle=True,
                        initial_epoch=0)
    model.save(OUTPUT_MODEL)
