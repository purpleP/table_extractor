import cv as cv
import numpy as np


def blocks(data, value):
    isvalue = np.concatenate(([value], np.equal(a, value).view(np.int8), [value]))
    return np.where(np.abs(np.diff(isvalue)) == 1)[0].reshape(-1, 2)


def find_non_min(array, axis):
    means = cv.reduce(array, axis, cv.REDUCE_AVG, dtype=cv.CV_32F).flatten()
    blocks = blocks(means, 255)
    centers = (blocks[:,0] + blocks[:,1] - 1) / 2
    mkslice = lambda s, e: (slice(None), slice(s, e))
    if axis:
        mkslice = lambda s, e: (slice(s, e), slice(None))
    return np.stack(array[mkslice(s, e)] for start, end in zip(centers, centers[1:]))


img = cv.imread('1.tiff', cv.IMREAD_GRAYSCALE)
_, wb = cv.threshold(cv.imread('1.tiff', cv.IMREAD_GRAYSCALE), 127, 255, cv.THRESH_BINARY_INV)
row_means = cv.reduce(wb, 1, cv.REDUCE_AVG, dtype=cv.CV_32F).flatten()
row_gaps = zero_runs(row_means)
row_cutpoints = (row_gaps[:,0] + row_gaps[:,1] - 1) / 2

bounding_boxes = []
for start, end in zip(row_cutpoints, row_cutpoints[1:]):
    line = img[start:end]
    line = img_gray_inverted[start:end]

    column_means = cv.reduce(line_gray_inverted, 0, cv.REDUCE_AVG, dtype=cv.CV_32F).flatten()
    column_gaps = zero_runs(column_means)
    column_gap_sizes = column_gaps[:,1] - column_gaps[:,0]
    column_cutpoints = (column_gaps[:,0] + column_gaps[:,1] - 1) / 2

    filtered_cutpoints = column_cutpoints[column_gap_sizes > 5]

    for xstart,xend in zip(filtered_cutpoints, filtered_cutpoints[1:]):
        bounding_boxes.append(((xstart, start), (xend, end)))

    visualize_vp("article_vp_%02d.png" % n, line, column_means, filtered_cutpoints)

result = img.copy()

for bounding_box in bounding_boxes:
    cv.rectangle(result, bounding_box[0], bounding_box[1], (255,0,0), 2)
