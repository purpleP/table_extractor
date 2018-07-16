from functools import reduce

import cv2 as cv
import numpy as np


def show(im, title='Image'):
    cv.imshow(title, im)
    cv.waitKey(0)


def consecutive(arr, axis):
    nonempty_rows = np.where(arr.any(axis=axis))[0]
    slices_ends = np.where(np.diff(nonempty_rows) > 1)[0]
    slices_starts = np.insert(slices_ends + 1, 0, 0)
    slices_ends = np.insert(slices_ends, len(slices_ends), len(nonempty_rows) - 1)
    return np.dstack((nonempty_rows[slices_starts], nonempty_rows[slices_ends]))[0]


def intervals(arr):
    return np.diff(np.hstack(arr)[1:-1].reshape(-1, 2)).flatten()


def join_rects(threshold, bounds):
    if len(bounds) <= 1:
        return bounds
    def join(acc, bounds):
        (s1, e1), (s2, e2) = acc[-1], bounds
        if s2 - e1 <= threshold:
            acc[-1] = (s1, e2)
        else:
            acc.append((s2, e2))
        return acc
    return np.array(reduce(join, bounds[1:], [bounds[0]]))


def find_aligned(bounds):
    pass


img = cv.imread('1.tiff', cv.IMREAD_GRAYSCALE)
_, wb = cv.threshold(cv.imread('1.tiff', cv.IMREAD_GRAYSCALE), 127, 255, cv.THRESH_BINARY_INV)
letters = [
    (start_end, consecutive(wb[slice(*start_end)], axis=0))
    for start_end in consecutive(wb, axis=1)
]
threshold = np.percentile(np.concatenate([intervals(xs) for _, xs in letters]), 95)
blocks = [(_, join_rects(threshold, xs)) for _, xs in letters]


for (y1, y2), xs in blocks:
    for x1, x2 in xs:
        cv.rectangle(wb, (x1, y1), (x2, y2), 255)
show(wb)
