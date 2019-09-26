import numpy as np
import scipy
import scipy.cluster
import cv2 as cv
import mss
from PIL import Image
import time
from multiprocessing import Queue, Pool
from mmcq import get_dominant_color
from median_cut import median_cut

def getDominantColor(image, num_clusters=3):
    shape = image.shape
    ar = image.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(ar, num_clusters)


    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

    index_max = scipy.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    return tuple(peak)

def benchmark(thread_count, func, patches):
    # Run multiple times to average results
    NB_OF_RUNS = 20
    avg_time = 0

    if thread_count == 1:
        for i in range(NB_OF_RUNS):
            start = time.time()
            for patch in patches:
                func(patch)
            end = time.time() - start
            #print('     >>> Run {} of {} took {} s'.format(i, func.__name__, end))
            avg_time =+ end
        avg_time / end
        print(' >>> ### AVERAGE OF {} TOOK {} s'.format(func.__name__, avg_time))
    else:
        for i in range(NB_OF_RUNS):
            pool = Pool(thread_count)
            # Create mapping for each image
            colors = pool.map(func, patches) 
            pool.close() 
            start = time.time()
            pool.join() 
            end = time.time() - start
            #print('     >>> Run {} of {} took {} s'.format(i, func.__name__, end)) 
            avg_time =+ end
        avg_time / end
        print(' >>> ### AVERAGE OF {} TOOK {} s'.format(func.__name__, avg_time))
        

def main():
    test_file = 'test.jpg'

    im = cv.imread(test_file)
    im_PIL = Image.open(test_file)
    im_PIL_np = np.array(im_PIL)

    h, w, _ = im.shape

    COL_SCAN = 8
    ROW_SCAN = 4

    kernel_col_size = h//ROW_SCAN
    kernel_row_size = w//COL_SCAN

    patches = []
    patches_PIL = []

    for i in range(ROW_SCAN):
        for j in range(COL_SCAN):
            x1 = (j*kernel_row_size)
            x2 = (kernel_row_size+x1)
            y1 = (i*kernel_col_size) 
            y2 = (kernel_col_size+y1) 

            if x1 == 0 or y1 == 0 or x2 == w or y2 == h:
                #color = getDominantColor(im[y1:y2, x1:x2, :])
                patches.append(im[y1:y2, x1:x2, :])
                patches_PIL.append(Image.fromarray(im_PIL_np[y1:y2, x1:x2, :]))

    for i in [4, 6, 8, 12, 16, 20]:
        print('Starting getDominantColor() with {} processes'.format(i))
        benchmark(i, getDominantColor, patches)

        print('Starting median_cut() with {} processes'.format(i))
        benchmark(i, median_cut, patches_PIL)

        print('------------------------------------------------------------------------------')

if __name__ == "__main__":
    main()