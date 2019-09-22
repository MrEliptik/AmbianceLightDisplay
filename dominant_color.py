import pyautogui
import numpy as np
import scipy
import scipy.cluster
import cv2 as cv
import mss
from PIL import Image
import time
from multiprocessing.pool import Pool
 
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
    # Take screenshot
    #im = pyautogui.screenshot() #pretty slow (100ms)
    with mss.mss() as sct:
        # Get rid of the first, as it represents the "All in One" monitor:
        while "Screen capturing":
            last_time = time.time()

            # Get raw pixels from the screen, save it to a Numpy array
            im = np.array(sct.grab(sct.monitors[2]))

            h, w, _ = im.shape

            im = cv.resize(im, (int(w/6), int(h/6)), cv.INTER_LINEAR)

            h, w, _ = im.shape

            COL_SCAN = 6
            ROW_SCAN = 4

            kernel_col_size = h//ROW_SCAN
            kernel_row_size = w//COL_SCAN

            patches = []

            for i in range(ROW_SCAN):
                for j in range(COL_SCAN):
                    x1 = j*kernel_row_size
                    y1 = i*kernel_col_size
                    x2 = kernel_row_size+(j*kernel_row_size)
                    y2 = kernel_col_size+(i*kernel_col_size)

                    if x1 == 0 or y1 == 0 or x2 == w or y2 == h:
                        #color = getDominantColor(im[y1:y2, x1:x2, :])
                        #print(color)
                        #cv.rectangle(im, (x1,y1), (x2,y2), color, 3)
                        patches.append(im[y1:y2, x1:x2, :])

            pool = Pool(8)
            # Create mapping for each image
            colors = pool.map(getDominantColor, patches) 
            pool.close() 
            pool.join()

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
