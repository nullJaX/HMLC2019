#!/usr/bin/env python3
import itertools
import json
import random
from threading import Thread

from tqdm import tqdm, tnrange

from sword import morphology_map, where_da_sword


def iou(expected, output):
    xA = max(expected[0], output[0])
    yA = max(expected[1], output[1])
    xB = min(expected[2], output[2])
    yB = min(expected[3], output[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    expectedArea = (expected[2] - expected[0] + 1) * (
            expected[3] - expected[1] + 1)
    outputArea = (output[2] - output[0] + 1) * (output[3] - output[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (float(expectedArea + outputArea - interArea) + 1.0)

    # return the intersection over union value
    return iou


def load_data(i):
    data = []
    with open("./train/out{0!s}.txt".format(i)) as f:
        for line in f:
            line = line.replace("\n", "")
            line = line.replace("#", "-1")
            line = line.split(";")
            line = tuple(map(int, line))
            data.append(line)
    return data


def validate(j, params, iou_for_params):
    expected = load_data(j)
    iou_for_file = []
    for i, frame in enumerate(where_da_sword("./train/in{0!s}.mp4".format(j), expected[0], params)):
        iou_for_file.append(iou(expected[i], frame))
    iou_for_params.append(sum(iou_for_file) / len(iou_for_file))


if __name__ == '__main__':
    with open("config.json", "r") as f:
        best_params = dict(json.load(f))
    best_iou = []
    threads = []
    for j in range(1, 9, 1):
        threads.append(
            Thread(target=validate, args=(j, best_params, best_iou)))
    for t in threads:
        t.start()
    for t in tqdm(threads):
        t.join()
    best_iou = sum(best_iou) / len(best_iou)
    print("Best IoU: {0!s}".format(best_iou))

    blur_kernel_size = [i for i in range(3, 9, 2)]
    canny_threshold1 = [i for i in range(30, 100, 15)]
    canny_threshold2 = [i for i in range(140, 250, 20)]
    hough_rho = [i / 100 for i in range(0, 120, 30)]
    hough_threshold = [i for i in range(0, 100, 15)]
    hough_minLineLength = [i for i in range(20, 100, 20)]
    hough_maxLineGap = [i for i in range(0, 24, 3)]
    morphology_method = [i for i in morphology_map.keys()]
    morphology_kernel = [i for i in range(3, 9, 2)]
    morphology_iter = [i for i in range(0, 4, 1)]

    for _ in tqdm(range(10000)):
        params = {
            "deque_maxlen": best_params["deque_maxlen"],
            "blur_kernel_size": random.choice(blur_kernel_size),
            "canny_threshold1": random.choice(canny_threshold1),
            "canny_threshold2": random.choice(canny_threshold2),
            "hough_rho": random.choice(hough_rho),
            "hough_threshold": random.choice(hough_threshold),
            "hough_minLineLength": random.choice(hough_minLineLength),
            "hough_maxLineGap": random.choice(hough_maxLineGap),
            "morphology_method": random.choice(morphology_method),
            "morphology_kernel": random.choice(morphology_kernel),
            "morphology_iter": random.choice(morphology_iter)
        }
        threads = []
        iou_for_params = []
        for j in range(1, 9, 1):
            threads.append(Thread(target=validate, args=(j, params, iou_for_params)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        iou_for_params = sum(
            iou_for_params) / len(
            iou_for_params)
        if iou_for_params > best_iou:
            best_params = params
            best_iou = iou_for_params
            print(
                "Better IoU: {0!s}".format(
                    best_iou))
            with open("config.json", "w") as f:
                json.dump(best_params, f)
