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
        yield data_file


def generator(data):
    while True:
        random.shuffle(data)
        for room in data:
            yield ([room[:, :, :, :, 0:3]], [room[:, :, :, :, 3:]])


def create_grid(data):
    rooms = []
    for room in data:
        n_bins = 75
        x_min, x_max = numpy.min(room[:, 0]), numpy.max(room[:, 0])
        y_min, y_max = numpy.min(room[:, 1]), numpy.max(room[:, 1])
        z_min, z_max = numpy.min(room[:, 2]), numpy.max(room[:, 2])
        room[:, 0] = numpy.digitize(room[:, 0], numpy.linspace(x_min, x_max, n_bins, dtype=numpy.int)) -1
        room[:, 1] = numpy.digitize(room[:, 1], numpy.linspace(y_min, y_max, n_bins, dtype=numpy.int)) -1
        room[:, 2] = numpy.digitize(room[:, 2], numpy.linspace(z_min, z_max, n_bins, dtype=numpy.int)) -1
        room_grid = numpy.zeros((n_bins, n_bins, n_bins, 4))
        for item in room[:]:
            room_grid[int(item[0]), int(item[1]), int(item[2]), :] = item[3:]
        rooms.append(numpy.expand_dims(room_grid, axis=0))
    return rooms


if __name__ == '__main__':
    data = load_data()
    data = create_grid(data)
    callbacks = init_callbacks()
    model = load_model(RAW_MODEL)
    # model = create_model()
    model.fit_generator(generator(data),
                        steps_per_epoch=4,
                        epochs=EPOCHS,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=generator(data),
                        validation_steps=1,
                        max_queue_size=10,
                        shuffle=True,
                        initial_epoch=0)
    model.save(OUTPUT_MODEL)
