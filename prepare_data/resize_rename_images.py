import cv2
import os
import glob
import tqdm
from datetime import date
import random

inputFolder = 'original_images'
outputFolder = 'supervisely_images'

if not os.path.exists(outputFolder):
    os.mkdir(outputFolder)

def resize_and_save():
    date_str = time_stamp()
    image_path_list = glob.glob(os.path.join(inputFolder, '*'))
    random.shuffle(image_path_list)
    i = 0
    for path in tqdm.tqdm(image_path_list):
        image = cv2.imread(path)
        resized_image = resize(image, 1024)
        cv2.imwrite( outputFolder+'\\pricetag_' + date_str +'_%i.jpeg' %i,resized_image)
        i+=1

def resize(image, max_length=1024):
    height, width = image.shape[:2]
    long_edge = width if width>height else height
    ratio = max_length / long_edge
    image = cv2.resize(image, (int(width*ratio), int(height*ratio)))
    return image

def time_stamp():
    today = date.today()
    date_str = today.strftime("%d_%m_%Y")
    return date_str
    
if __name__ == "__main__":
    resize_and_save()
