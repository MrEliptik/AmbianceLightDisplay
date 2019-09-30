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

## FLAGS
# Full speed if -1
FIXED_FPS = 5
NB_MONITOR = 1
MONITOR_1 = {'id':1, 'w':2560, 'h':1080, 'col':6, 'row':4,'active':True}
MONITOR_2 = {'id':2, 'w':3840, 'h':2160, 'col':8, 'row':4, 'active':False}
MONITORS = [MONITOR_1, MONITOR_2]
#MONITOR_1 = {'id':1, 'w':1920, 'h':1080, 'col':6, 'row':4,'active':True}
#MONITORS = [MONITOR_1]
 
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
        patch_id, frame = input_q.get()
        if (frame is not None):
            color = getDominantColor(frame)
            output_q.put((patch_id, color))

def grabber(input_q, output_q):
    with mss.mss() as sct:
        while True:
            for monitor in MONITORS:
                if not monitor['active']:
                    continue
                ready = input_q.get()
                if ready is not None:
                    # Get raw pixels from the screen, save it to a Numpy array
                    im = np.array(sct.grab(sct.monitors[monitor['id']])) # ~35ms on 4k display, 20ms 1080p
                    h, w, _ = im.shape
                    im = cv.resize(im, (int(w/6), int(h/6)), cv.INTER_LINEAR)
                    output_q.put(im)

def main(display=False, debug="none"):
    if debug == "debug":
        debug = True
        timing = True
    elif debug == "timing":
        timing = True
        debug = False
    else:
        timing = False
        debug = False

    if debug:
        print("Creating queues..")
    input_q = Queue()
    output_q = Queue()
    grabber_in_q = Queue()
    grabber_out_q = Queue()

    if debug:
        print('Spinning up workers..')
    # Spin up workers to parallelize workload
    pool = Pool(8, worker, (input_q, output_q))
    grabber_pool = Pool(1, grabber, (grabber_in_q, grabber_out_q))

    # Initial capture
    grabber_in_q.put("go")

    # Get rid of the first, as it represents the "All in One" monitor:
    while "Screen capturing":
        if debug:
            print('Starts grabbing..')
        last_time = time.time()

        if timing:
            last_screen = time.time()

        for monitor in MONITORS:
            if not monitor['active']:
                continue

            PATCHES_NB = ((monitor['col']+monitor['row'])*2) - 4
            patches = [None]*PATCHES_NB
            colors = [None]*PATCHES_NB
            
            # Get raw pixels from the screen, save it to a Numpy array
            #im = np.array(sct.grab(sct.monitors[monitor['id']])) # ~35s on 4k display, 20ms 1080p
            im = grabber_out_q.get()
            
            if timing:
                print("Screen time: {} ms".format((time.time() - last_screen)*1000))

            last_time_delay = time.time()


            h, w, _ = im.shape
            if debug:
                print('h: {}, w: {}'.format(h, w))

            kernel_col_size = h//monitor['row']
            kernel_row_size = w//monitor['col']

            if timing:
                calc_time = time.time()
                patch_extract_time = time.time()
            if debug:
                print('kernel size: {},{}'.format(kernel_col_size, kernel_row_size))
                print('patch size: {}'.format(PATCHES_NB))
                print('Starts processing screenshot')
                l = 0
            _id = 0 

            # Processing is starting, we can
            # grab another frame
            grabber_in_q.put('go')
            for i in range(monitor['row']):
                for j in range(monitor['col']):
                    x1 = (j*kernel_row_size)
                    x2 = (kernel_row_size+x1)
                    y1 = (i*kernel_col_size) 
                    y2 = (kernel_col_size+y1) 

                    if debug:
                        print('Patches: {}'.format(l))
                        print('Coords..: {}, {}, {}, {}'.format(x1,y1,x2,y2))
                        l += 1
                    if x1 == 0 or y1 == 0 or x2 == w or y2 == h:
                        input_q.put((_id, im[y1:y2, x1:x2, :]))
                        _id += 1
                        if debug:
                            print('Input q full: {}'.format(input_q.empty()))
                            print('Input q size: {}'.format(_id))
            if timing:
                print('Patch extraction: {} ms'.format((time.time() - patch_extract_time)*1000))

            if timing:
                last_process_finished = time.time()   
            k = 0
            while(k < PATCHES_NB): 
                if debug:
                    print('Waiting for patches {}'.format(k))
                patch_id, patch = output_q.get()
                if timing:
                    print('Process {} finished: {} ms'.format(patch_id, (time.time() - last_process_finished)*1000))
                colors[patch_id] = patch
                k += 1
            
            if timing:
                print('Last process finished: {} ms'.format((time.time() - last_process_finished)*1000))
                print('calc time: {} ms'.format((time.time() - calc_time)*1000))

            if debug:
                print(colors)

            if display:
                _id = 0
                for i in range(monitor['row']):
                    for j in range(monitor['col']):
                        x1 = (j*kernel_row_size)
                        x2 = (kernel_row_size+x1)
                        y1 = (i*kernel_col_size) 
                        y2 = (kernel_col_size+y1) 

                        if x1 == 0 or y1 == 0 or x2 == w or y2 == h:
                            input_q.put((_id, im[y1:y2, x1:x2, :]))
                            cv.rectangle(im, (x1,y1), (x2,y2), colors[_id])
                            _id += 1
                    
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
    main(display=True, debug="none")
