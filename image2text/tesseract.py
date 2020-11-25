import cv2
import pytesseract
import numpy as np

import tqdm
from os import listdir
from os.path import isfile, join
import os 
import glob 

# test
MODEL_NAME = "test_results"
IMAGE_DIR = "test_images"

# MODEL_NAME = "test2"
# IMAGE_DIR = "../data/model_outputs/" + MODEL_NAME + "_cropped"
OUT_DIR = "outputs/" + MODEL_NAME + "_2"

psm = '8'

if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

image_file_names = [f for f in listdir(IMAGE_DIR) if isfile(join(IMAGE_DIR, f))]
image_file_names.sort()

for image_file_name in tqdm.tqdm(image_file_names):
    image_path = os.path.join(IMAGE_DIR, image_file_name)

    # print(image_file_name)
    image = cv2.imread(image_path).astype(np.uint8)
    # image = cv2.resize(image, (224,224))
    # image = cv2.resize(image, None, fx=2, fy=2)


    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_h, img_w = gray_image.shape
    gray, thresh = cv2.threshold(gray_image,100,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # kernel = np.ones((2, 1), np.uint8)
    # img = cv2.erode(gray, kernel, iterations=1)
    # # cv2.imshow("asd",img)
    # # cv2.waitKey(0) 
    # # img = cv2.dilate(img, kernel, iterations=1)
    # # cv2.imshow("asd",img)
    # # cv2.waitKey(0) 

    

    custom_config = r'-l tur --oem 3 --psm ' + psm #+ 'outbase digits'
    out_below = pytesseract.image_to_string(thresh, config=custom_config)

    boxes = pytesseract.image_to_boxes(thresh)
    thresh_3_channel = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    for b in boxes.splitlines():
        # print(b)
        b = b.split(' ')
        print(image_file_name)
        print(b)
        x,y,w,h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        cv2.rectangle(thresh_3_channel,(x,img_h-y),(w,img_h-h),(0,0,255),1)
        cv2.putText(thresh_3_channel,b[0],(x+10,img_h-y-10),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),2)

    cv2.imwrite(OUT_DIR + "/" + image_file_name, thresh_3_channel)


    # print("OUTPUT:", repr(out_below))
    print(image_file_name)
    print("OUTPUT:", out_below)

    
    # images = np.hstack((image,grey_3_channel))
    # image_array = ([image,thresh])
    

    # cv2.imshow(image_file_name,thresh_3_channel)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()

    # image_text = out_below.replace('\n','')
    # image_text = image_text.replace('\x0c','')

    # blk = np.zeros(image.shape, np.uint8)
    # cv2.rectangle(blk, (0, 0), (image.shape[1], 40), (255, 255, 255), cv2.FILLED)
    # # Generate result by blending both images (opacity of rectangle image is 0.25 = 25 %)
    # out = cv2.addWeighted(image, 1.0, blk, 0.5, 1)

    # cv2.putText(out, "result: {}".format(image_text),
        # (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.imshow("Rotated", image)
    # cv2.imwrite(OUT_DIR + "/" + image_file_name, out)
   
