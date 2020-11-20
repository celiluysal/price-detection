import cv2
import os
import glob
import tqdm

inputFolder = 'orginal_images'
outputFolder = 'supervisely_images'
date = "20_11_20"


def resize_and_save():
    i = 0
    for img in tqdm.tqdm(glob.glob(inputFolder + "/*g")):
        image = cv2.imread(img)
        imgResized = cv2.resize(image,(1024,768))
        cv2.imwrite( outputFolder+'/pricetag_' + date +'_%i.jpg' %i,imgResized)
        i+=1


if __name__ == "__main__":
    resize_and_save()