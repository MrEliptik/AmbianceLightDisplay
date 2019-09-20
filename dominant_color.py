import pyautogui
import numpy as np
import scipy
import scipy.cluster
import cv2 as cv
 
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
    im = pyautogui.screenshot()

    print(type(im))

    im = np.array(im)

    h, w, _ = im.shape
    print(h, w)

    im = cv.resize(im, (int(h/4), int(w/4)), cv.INTER_LINEAR)
    h, w, _ = im.shape
    print(h, w)

    colors = getDominantColor(im)
    print(colors)
    
    # Save the image
    #pic.save('Screenshot.png') 

if __name__ == "__main__":
    main()
