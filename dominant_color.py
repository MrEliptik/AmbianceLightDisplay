import pyautogui
import numpy as np
import scipy
import scipy.cluster
import cv2 as cv
import mss
from PIL import Image
import time
from multiprocessing import Queue, Pool
import sys
 
def getDominantColor(image, num_clusters=3):
    shape = image.shape
    ar = image.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(ar, num_clusters)


    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

    index_max = scipy.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    return tuple(peak)

def worker(input_q, output_q):
    while True:
        #print("> ===== in worker loop, frame ", frame_processed)
        patch_id, frame = input_q.get()
        if (frame is not None):
            color = getDominantColor(frame)
            output_q.put((patch_id, color))

def main(display=False, debug=False):
    ## FLAGS
    # Full speed if -1
    FIXED_FPS = 10

    COL_SCAN = 8
    ROW_SCAN = 4

    PATCHES_NB = ((COL_SCAN+ROW_SCAN)*2) - 4

    NB_MONITOR = 1
    MONITOR_1 = {'id':1, 'w':2560, 'h':1080, 'active':False}
    MONITOR_2 = {'id':2, 'w':3840, 'h':2160, 'active':True}
    MONITORS = [MONITOR_1, MONITOR_2]

    patches = [None]*PATCHES_NB
    colors = [None]*PATCHES_NB

    if debug:
        print("Creating queues..")
    input_q = Queue()
    output_q = Queue()

    if debug:
        print('Spinning up workers..')
    # Spin up workers to parallelize workload
    pool = Pool(16, worker, (input_q, output_q))

    
    # Take screenshot
    with mss.mss() as sct:
        # Get rid of the first, as it represents the "All in One" monitor:
        while "Screen capturing":
            if debug:
                print('Starts grabbing..')
            last_time = time.time()

            if debug:
                last_screen = time.time()

            for monitor in MONITORS:
                if not monitor['active']:
                    continue
                # Get raw pixels from the screen, save it to a Numpy array
                im = np.array(sct.grab(sct.monitors[monitor['id']])) # ~35s on 4k display, 20ms 1080p
                if debug:
                    print("Screen time: {} ms".format((time.time() - last_screen)*1000))

                last_time_delay = time.time()

                h, w, _ = im.shape

                im = cv.resize(im, (int(w/6), int(h/6)), cv.INTER_LINEAR)

                h, w, _ = im.shape

                kernel_col_size = h//ROW_SCAN
                kernel_row_size = w//COL_SCAN

                if debug:
                    print('kernel size: {},{}'.format(kernel_col_size, kernel_row_size))

                # size is == to rectangle's perimeter
                
                if debug:
                    print('patch size: {}'.format(PATCHES_NB))
                    print('Starts processing screenshot')
                u = 0 
                l = 0 
                for i in range(ROW_SCAN):
                    for j in range(COL_SCAN):
                        x1 = (j*kernel_row_size)
                        x2 = (kernel_row_size+x1)
                        y1 = (i*kernel_col_size) 
                        y2 = (kernel_col_size+y1) 

                        l += 1
                        if debug:
                            print('Patches: {}'.format(l))
                            print('Coords..: {}, {}, {}, {}'.format(x1,y1,x2,y2))
                        if x1 == 0 or y1 == 0 or x2 == w or y2 == h:
                            input_q.put((u, im[y1:y2, x1:x2, :]))
                            u += 1
                            if debug:
                                print('Input q full: {}'.format(input_q.empty()))
                                print('Input q size: {}'.format(u))
                            
                k = 0
                while(k < PATCHES_NB): 
                    if debug:
                        print('Waiting for patches {}'.format(k))
                    patch_id, patch = output_q.get()
                    colors[patch_id] = patch
                    k += 1

                if debug:
                    print(colors)

                if display:
                    cv.putText(im, "delay (ms): {}".format(int(1 / (time.time() - last_time_delay))), 
                        (0, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (200, 100, 170), 
                        3, lineType=cv.LINE_AA)

                print("fps: {}".format(1 / (time.time() - last_time)))

                # Press "q" to quit
                if cv.waitKey(1) & 0xFF == ord("q"):
                    cv.destroyAllWindows()
                    sys.exit()

                while((time.time() - last_time) < (1/FIXED_FPS)):
                    if debug:    
                        print("sleep")
                    #sleep the remaining time
                    #time.sleep((1/FIXED_FPS) - time.time())
                    time.sleep(0.01)

                if display:
                    # Display fps on the image
                    cv.putText(im, "fps: {}".format(int(1 / (time.time() - last_time))), 
                        (0, 25), cv.FONT_HERSHEY_SIMPLEX, 1.0, (200, 100, 170), 
                        3, lineType=cv.LINE_AA)

                    # Display the picture
                    cv.imshow("OpenCV/Numpy normal", im)

if __name__ == "__main__":
    main(display=True)
