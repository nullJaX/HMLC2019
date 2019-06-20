#!/usr/bin/env python3
import random
import sys

import numpy
from keras.engine.saving import load_model

from config import OUTPUT_MODEL, BATCH_SIZE, RAW_MODEL, N_SAMPLES


def load_data(file):
    data = []
    with open(file) as fin:
        for line in fin:
            x, y, z, r, g, b = map(float, line.split())
            r /= 255
            g /= 255
            b /= 255
            data.append([x, y, z, r, g, b])
    orig_len = len(data)
    while len(data) % BATCH_SIZE != 0 or len(data) % N_SAMPLES != 0:
        data.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    random.shuffle(data)
    return data, orig_len

def generator(data):
    data = numpy.array(data)
    while True:
        batch_location = []
        batch_color = []
        for i in range(0, len(data), N_SAMPLES):
            if len(batch_location) == BATCH_SIZE or len(batch_color) == BATCH_SIZE:
                yield ([batch_location, batch_color])
                batch_location = []
                batch_color = []
            batch = data[i:i+N_SAMPLES, :]
            batch_location.append(batch[:, 0:3])
            batch_color.append(batch[:, 3:6])


if __name__ == '__main__':
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    data, orig_len = load_data(in_file)
    model = load_model(RAW_MODEL)
    output = model.predict_generator(generator(data),
                                     steps=int(numpy.ceil(len(data) / (BATCH_SIZE*N_SAMPLES))),
                                     max_queue_size=10,
                                     workers=1,
                                     use_multiprocessing=False,
                                     verbose=0)
    with open(out_file, "w") as fout:
        i = 0
        for o in output:
            for sample in o:
                if i >= orig_len:
                    break
                else:
                    fout.write(str(int(numpy.round(sample))) + "\n")
                    i += 1
