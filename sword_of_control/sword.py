#!/usr/bin/env python3
import sys
import json
from collections import deque

import cv2
import numpy as np

morphology_map = {"dilate": cv2.MORPH_DILATE,
                  "close": cv2.MORPH_CLOSE,
                  "erode": cv2.MORPH_ERODE,
                  "open": cv2.MORPH_OPEN}


def distance(last, current):
    cxa, cya, cxb, cyb = current
    cxc = int(np.round((cxa + cxb) / 2))
    cyc = int(np.round((cya + cyb) / 2))
    xa, ya, xb, yb, xc, yc = last
    dist_a = np.sqrt((xa + cxa) ** 2 + (ya + cya) ** 2)
    dist_b = np.sqrt((xb + cxb) ** 2 + (yb + cyb) ** 2)
    dist_c = np.sqrt((xc + cxc) ** 2 + (yc + cyc) ** 2)
    return (dist_a + dist_b + dist_c) / 3


def compare(last_frame, frame, last_positions, params):
    # difference between frames
    frame_diff = cv2.absdiff(frame, last_frame)
    frame_diff = cv2.morphologyEx(frame_diff, morphology_map[params["morphology_method"]], (params["morphology_kernel"], params["morphology_kernel"]), iterations=params["morphology_iter"])

    # Limit Region Of Interest by creating circle of radius equal to longer dimension delta
    mask = np.zeros(frame_diff.shape, dtype=np.uint8)
    for roi in last_positions:
        radius = max(abs(roi[3] - roi[1]), abs(roi[2] - roi[0]))
        cv2.circle(mask, roi[-2:], radius, (255, 255, 255), -1, 8, 0)
    frame_diff = frame_diff & mask
    # cv2.imshow("CannyEdges", frame_diff)

    # Detect edges
    canny = cv2.Canny(frame_diff, params["canny_threshold1"], params["canny_threshold2"])
    lines = cv2.HoughLinesP(canny, params["hough_rho"], np.pi / 180,
                            params["hough_threshold"],
                            minLineLength=params["hough_minLineLength"],
                            maxLineGap=params["hough_maxLineGap"])
    # frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_GRAY2BGR)
    # cv2.imshow("CannyEdges", canny)


    # Determine new points
    if lines is not None:
        distances = [distance(last_positions[-1], line[0]) for line in lines]
        index = distances.index(min(distances))
        xa, ya, xb, yb = lines[index][0]
        xc = int(np.round((xa + xb) / 2))
        yc = int(np.round((ya + yb) / 2))
    else:
        xa, ya, xb, yb, xc, yc = last_positions[-1]
    return xa, ya, xb, yb, xc, yc


def preprocess(frame, params):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.medianBlur(frame_gray, params["blur_kernel_size"])
    return frame_blur


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
    x_c = int(np.round(sum(first_frame_position[0::2]) / 2))
    y_c = int(np.round(sum(first_frame_position[1::2]) / 2))
    last_positions = deque([(*first_frame_position, x_c, y_c)], maxlen=params["deque_maxlen"])
    last_frame = None
    for i, frame in enumerate(mp4_read(video_file_path)):
        if i == 0:
            last_frame = preprocess(frame, params)
        else:
            frame = preprocess(frame, params)
            last_positions.append(tuple(map(int, compare(last_frame, frame, last_positions, params))))
            last_frame = frame.copy()
        x_a, y_a, x_b, y_b, x_c, y_c = last_positions[-1]
        height, width = last_frame.shape[:2]
        x_a = x_a if 0 <= x_a < width else -1
        x_b = x_b if 0 <= x_b < width else -1
        y_a = y_a if 0 <= y_a < height else -1
        y_b = y_b if 0 <= y_b < height else -1
        display = cv2.cvtColor(last_frame, cv2.COLOR_GRAY2RGB)
        if all([i >= 0 for i in [x_a, x_b, y_a, y_b]]):
            cv2.line(display, (x_a, y_a), (x_b, y_b), (0, 255, 0), 10)
        cv2.circle(display, (x_a, y_a), 5, (255, 0, 0), -1)
        cv2.circle(display, (x_b, y_b), 5, (0, 0, 255), -1)
        cv2.circle(display, (x_c, y_c), 5, (0, 0, 0), -1)
        cv2.imshow('Debug display', display)
        cv2.waitKey(30)
        yield x_a, y_a, x_b, y_b


if __name__ == "__main__":
    video_file_path = sys.argv[1]
    first_frame_position = tuple(map(int, sys.argv[2].split(";")))
    with open("config.json", "r") as f:
        params = dict(json.load(f))
    for x_a, y_a, x_b, y_b in where_da_sword(video_file_path,
                                             first_frame_position, params):
        if x_a < 0 or y_a < 0:
            x_a = "#"
            y_a = "#"
        if x_b < 0 or y_b < 0:
            x_b = "#"
            y_b = "#"
        print("{0!s};{1!s};{2!s};{3!s}".format(x_a, y_a, x_b, y_b))
    print()
