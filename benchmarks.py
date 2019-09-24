import numpy as np
import scipy
import scipy.cluster
import cv2 as cv
import mss
from PIL import Image
import time
from multiprocessing import Queue, Pool
from mmcq import get_dominant_color

def getDominantColor(image, num_clusters=3):
    shape = image.shape
    ar = image.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(ar, num_clusters)


    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

    index_max = scipy.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    return tuple(peak)

def main():
    test_file = 'test.jpg'

    im = cv.imread(test_file)

    h, w, _ = im.shape

    start = time.time()
    color = getDominantColor(im)
    print('getDominantColor on image of size: ({},{}) took {}s'.format(h, w, time.time() - start))

    start = time.time()
    color = get_dominant_color(filename=test_file)
    print('mmcq.get_dominant_color on image of size: ({},{}) took {}s'.format(h, w, time.time() - start))

if __name__ == "__main__":
    main()