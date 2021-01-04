import glob, os, cv2
from os.path import join
import numpy as np


#PARAMETERS
IMG_PROCESS_HEIGHT = 2000
IMG_CHRACTER_HEIGHT = 1000
SPACE_RATE = 5
IGNORE_CH_PERCENT = 40
CURSOR_H_PERCENT = 0.25
PADDING_V = 0.2
PADDING_H = 0.3

# name = "7"

# INPUT_FILE = "input1"
# OUTPUT_FILE = "results\\result"+name


# if not os.path.exists(OUTPUT_FILE):
#     os.mkdir(OUTPUT_FILE)

def get_input_path_list():
    image_path_list = glob.glob(os.path.join(INPUT_FILE, '*'))
    # image_path_list.sort()
    return image_path_list

def read_all(path_list):
    images = []
    for file in path_list:
        image = cv2.imread(file)
        images.append(image)
    return images
               
def save_all(images, path_list):
    for file, image in zip(path_list, images): 
        image_name = (file).split('\\')[-1].split('.')[0]
        saving_file = join(OUTPUT_FILE, image_name + ".jpeg")
        cv2.imwrite(saving_file, image) 
    
def run():
    path_list = get_input_path_list()
    images = read_all(path_list)
        
    result_images = []
    for img in images:
        img = resize(img, IMG_PROCESS_HEIGHT)
        # img = preprocess(img)
        # img, number_image_list = find_nummbers(img)
        number_image_list = find_nummbers(img)
        if number_image_list:
            print("number_image_list",len(number_image_list))
        else:
            print("boş")
            # for im in number_image_list:
                # save_all(result_images, path_list) 
        
        result_images.append(img)
    
    save_all(result_images, path_list)  
    
def save_images(image_list, file_name):
    counter = 0
    for img in image_list:
        saving_file = file_name+ "\\"+str(counter)+ ".jpeg"
        cv2.imwrite(saving_file ,img)
        counter = counter + 1
    
def get_numbers(image):
    image = resize(image)
    number_image_list = find_nummbers(image)
    return number_image_list
    
def resize(image, max_height=3000):
    height, width = image.shape[0], image.shape[1]
    rate = height / max_height
    img = cv2.resize(image,(int(width/rate), int(height/rate)))
    return img
    
def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    kernel = np.ones((10,10),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    
    kernel = np.ones((10,10),np.uint8)
    dilate = cv2.dilate(opening,kernel,iterations=3)
    
    blur = cv2.blur(dilate,(20,20))
    ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return thresh

def find_nummbers(image):
    thresh = preprocess(image)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangle_list = []
    number_image_list = []
    height, width= image.shape[0], image.shape[1]
    
    spc = int(height * SPACE_RATE / 1000)
    cursor_h = int(height*CURSOR_H_PERCENT)
        
    if contours:
        for i in range(len(contours)):
            x,y,w,h = cv2.boundingRect(contours[i])
            x1, y1, x2, y2 = limit_edges(width, height,
                                          x-spc, y-spc, x+w+spc, y+h+spc)
            coords = [(x1, y1), (x2, y2)]
            rectangle_list.append(coords)
                 
        rectangle_list = select_and_sort(cursor_h, rectangle_list)
        number_image_list = crop_numbers(thresh,rectangle_list)
    
    return number_image_list

def limit_edges(w, h, x1, y1, x2, y2):
    x1 = 0 if x1 < 0 else x1
    y1 = 0 if y1 < 0 else y1
    x2 = w if x2 > w else x2
    y2 = h if y2 > h else y2
    return x1, y1, x2, y2

def select_and_sort(cursor, rectangle_list):
    select_list = []
    for rectangle in rectangle_list:
        x1, y1 = rectangle[0]
        x2, y2 = rectangle[1]
        if(y1<cursor and cursor<y2):
            select_list.append(rectangle)
    return sorted(select_list, key=lambda x: x[0][0])

def crop_numbers(image, rectangle_list):
    number_image_list = []
    
    for rectangle in rectangle_list:
        x1, y1 = rectangle[0]
        x2, y2 = rectangle[1]
        
        crop_image = image[y1:y2, x1:x2]
        crop_image = resize(crop_image, IMG_CHRACTER_HEIGHT)
        contour = clean_arraund_number(crop_image)
        img = add_padding(contour, PADDING_V, PADDING_H)
        number_image_list.append(img)

    return number_image_list

def add_padding(image, padding_v, padding_h):
    ht, wd = image.shape[0], image.shape[1]

    ww = int(wd + wd * padding_h * 2)
    hh = int(ht + ht * padding_v * 2)
    result = np.zeros((hh,ww), dtype=np.uint8)
    
    xx = (ww - wd) // 2
    yy = (hh - ht) // 2
    
    result[yy:yy+ht, xx:xx+wd] = image
    return result
    
    

def clean_arraund_number(thresh):
    height,width=thresh.shape
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        rectangle = cv2.boundingRect(cnt)
        x,y,w,h = rectangle
        cnt_area = w*h
        image_area = width * height
        area_percentence = cnt_area * 100 / image_area

        if (check_boundary(width,height,rectangle) and (area_percentence < IGNORE_CH_PERCENT)):
            thresh = cv2.drawContours(thresh, [cnt], -1, 0, -1)
    
    return thresh

def check_boundary(width,height,rectangle):
    x1,y1,w,h = rectangle
    x2,y2 = x1+w, y1+h
    if x1<=0 or y1<=0 or x2>=width-1 or y2>=height-1:
        return True 
    else: 
        return False
    

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


# if  __name__ == "__main__":   
#     image = cv2.imread(INPUT_FILE+"\\"+ name +".jpeg")
#     number_image_list = get_numbers(image)
#    # save_images(number_image_list, OUTPUT_FILE)
   
