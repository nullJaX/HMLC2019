#!/usr/bin/env python3
EPOCHS = 10000
BATCH_SIZE = 32
N_SAMPLES = 512
ITEMS = 953511 / 1000
VALIDATION_SPLIT = 0.35
INPUT_SHAPE = (75, 75, 75, 3)
OUTPUT_FEATURES = 1

RAW_MODEL = "pointnet_raw.hdf5"
OUTPUT_MODEL = "model.hdf5"
