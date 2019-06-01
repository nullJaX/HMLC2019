#!/usr/bin/env python3
import sys
import json
import cv2
import numpy as np


def compare(last_frame, frame, last_position, params):
    # TODO: Implement
    return last_position


def preprocess(frame, params):
    # TODO: Implement
    return frame


def mp4_read(video_file_path):
    cap = cv2.VideoCapture(video_file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fc = 0
    ret = True
    while fc < frame_count and ret:
        ret, buffer = cap.read()
        fc += 1
        yield buffer
    cap.release()


def where_da_sword(video_file_path, first_frame_position, params):
    last_position = first_frame_position
    last_frame = None
    for i, frame in enumerate(mp4_read(video_file_path)):
        if i == 0:
            last_frame = preprocess(frame, params)
        else:
            frame = preprocess(frame, params)
            last_position = tuple(map(int, compare(last_frame, frame,
                                                   last_position, params)))
            last_frame = frame
        x_a, y_a, x_b, y_b = last_position
        height, width = last_frame.shape[:2]
        x_a = x_a if 0 <= x_a < width else -1
        x_b = x_b if 0 <= x_b < width else -1
        y_a = y_a if 0 <= y_a < height else -1
        y_b = y_b if 0 <= y_b < height else -1
        yield x_a, y_a, x_b, y_b


if __name__ == "__main__":
    video_file_path = sys.argv[1]
    first_frame_position = tuple(map(int, sys.argv[2].split(";")))
    # with open("config", "r") as f:
    # params = dict(json.load(f))
    params = None
    for x_a, y_a, x_b, y_b in where_da_sword(video_file_path,
                                             first_frame_position, params):
        if x_a < 0 or y_a < 0:
            x_a = "#"
            y_a = "#"
        if x_b < 0 or y_b < 0:
            x_b = "#"
            y_b = "#"
        print("{0!s};{1!s};{2!s};{3!s}".format(x_a, y_a, x_b, y_b))
    print("")
