import pyautogui
import numpy as np
import scipy
import scipy.cluster
import cv2 as cv
import mss
from PIL import Image
import time
from multiprocessing import Queue, Pool
 
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
        frame = input_q.get()
        if (frame is not None):
            
            output_q.put(frame)

def main():
    #print("Creating queues..")
    input_q = Queue()
    output_q = Queue()

    #print('Spinning up workers..')
    # Spin up workers to parallelize workload
    pool = Pool(16, worker, (input_q, output_q))

    # Take screenshot
    #im = pyautogui.screenshot() #pretty slow (100ms)
    with mss.mss() as sct:
        # Get rid of the first, as it represents the "All in One" monitor:
        while "Screen capturing":
            #print('Starts grabbing..')
            last_time = time.time()

            # Get raw pixels from the screen, save it to a Numpy array
            im = np.array(sct.grab(sct.monitors[2]))

            h, w, _ = im.shape

            im = cv.resize(im, (int(w/6), int(h/6)), cv.INTER_LINEAR)

            h, w, _ = im.shape

            COL_SCAN = 8
            ROW_SCAN = 4

            PATCHES_NB = ((COL_SCAN+ROW_SCAN)*2) - 4

            kernel_col_size = h//ROW_SCAN
            kernel_row_size = w//COL_SCAN

            #print('kernel size: {},{}'.format(kernel_col_size, kernel_row_size))

            # size is == to rectangle's perimeter
            patches = [None]*PATCHES_NB
            #print('patch size: {}'.format(PATCHES_NB))

            #print('Starts processing screenshot')

            u = 0 
            l = 0 
            for i in range(ROW_SCAN):
                for j in range(COL_SCAN):
                    if j == 0:
                        x1 = (j*kernel_row_size)
                        x2 = kernel_row_size+x1
                    else:
                        x1 = (j*kernel_row_size)
                        x2 = (kernel_row_size+x1)
                    if i == 0:
                        y1 = i*kernel_col_size
                        y2 = kernel_col_size+y1
                    else:
                        y1 = (i*kernel_col_size) 
                        y2 = (kernel_col_size+y1) 

                    l += 1
                    #print('Patches: {}'.format(l))
                    #print('Coords..: {}, {}, {}, {}'.format(x1,y1,x2,y2))
                    if x1 == 0 or y1 == 0 or x2 == w or y2 == h:
                        #color = getDominantColor(im[y1:y2, x1:x2, :])
                        #print(color)
                        #cv.rectangle(im, (x1,y1), (x2,y2), color, 3)
                        #patches.append(im[y1:y2, x1:x2, :])
                        input_q.put(im[y1:y2, x1:x2, :])
                        u += 1
                        #print('Input q full: {}'.format(input_q.empty()))
                        #print('Input q size: {}'.format(u))
                        
            k = 0
            #while(k < PATCHES_NB and not output_q.empty()):
            while(k < PATCHES_NB): 
                #print('Waiting for patches {}'.format(k))
                patches[k] = output_q.get()
                k += 1

            '''
            pool = Pool(8)
            # Create mapping for each image
            colors = pool.map(getDominantColor, patches) 
            pool.close() 
            pool.join()
            '''

            # Display fps on the image
            cv.putText(im, "fps: {}".format(int(1 / (time.time() - last_time))), 
                (0, 25), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 
                3, lineType=cv.LINE_AA)

            # Display the picture
            cv.imshow("OpenCV/Numpy normal", im)

            #colors = getDominantColor(im)
            #print(colors)

            print("fps: {}".format(1 / (time.time() - last_time)))

            # Press "q" to quit
            if cv.waitKey(25) & 0xFF == ord("q"):
                cv.destroyAllWindows()
                break

    '''
    print(type(im))

    im = np.array(im)

    h, w, _ = im.shape
    print(h, w)

    im = cv.resize(im, (int(h/4), int(w/4)), cv.INTER_LINEAR)
    h, w, _ = im.shape
    print(h, w)

    colors = getDominantColor(im)
    print(colors)

    cv.imshow("screen", im)
    cv.waitKey(0)
    
    # Save the image
    #pic.save('Screenshot.png') 
    '''

if __name__ == "__main__":
    main()
