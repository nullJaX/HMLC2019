#!/usr/bin/env python3
import itertools
import json
import random
from threading import Thread

from tqdm import tqdm, tnrange

from sword import morphology_map, where_da_sword


def iou(a, b, epsilon=1e-5):
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
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

    blur_kernel_size = [i for i in range(3, 6, 2)]
    canny_threshold1 = [i for i in range(-1, 2, 1)]
    canny_threshold2 = [i for i in range(-1, 2, 1)]
    hough_rho = [i / 10000.0 for i in range(-100, 101, 10)]
    hough_threshold = [i for i in range(-1, 2, 1)]
    hough_minLineLength = [i for i in range(-1, 2, 1)]
    hough_maxLineGap = [i for i in range(-1, 2, 1)]
    morphology_method = [i for i in morphology_map.keys()]
    morphology_kernel = [i for i in range(3, 8, 2)]
    morphology_iter = [i for i in range(0, 3, 1)]

    for _ in tqdm(range(10000)):
        params = {
            "deque_maxlen": best_params["deque_maxlen"],
            "blur_kernel_size": random.choice(blur_kernel_size),
            "canny_threshold1": best_params["canny_threshold1"] + random.choice(canny_threshold1),
            "canny_threshold2": best_params["canny_threshold2"] + random.choice(canny_threshold2),
            "hough_rho": best_params["hough_rho"] + random.choice(hough_rho),
            "hough_threshold": best_params["hough_threshold"] + random.choice(hough_threshold),
            "hough_minLineLength": best_params["hough_minLineLength"] + random.choice(hough_minLineLength),
            "hough_maxLineGap": best_params["hough_maxLineGap"] + random.choice(hough_maxLineGap),
            "morphology_method": best_params["morphology_method"],
            "morphology_kernel": random.choice(morphology_kernel),
            "morphology_iter": random.choice(morphology_iter)
        }
        # print(params["hough_maxLineGap"])
        # print(params["hough_minLineLength"])
        # print(params["hough_rho"])
        # print(params["hough_threshold"])
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
