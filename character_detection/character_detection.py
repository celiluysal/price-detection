import glob, os, cv2, imutils
from os.path import join
import numpy as np


INPUT_FILE = "input"
OUTPUT_FILE = "output"


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
    
def resize(image, max_height=3000):
    height, width = image.shape[0], image.shape[1]
    rate = height / max_height
    img = cv2.resize(image,(int(width/rate), int(height/rate)))
    return img

def run():
    path_list = get_input_path_list()
    images = read_all(path_list)
        
    result_images = []
    for img in images:
        img = resize(img)
        # img = preprocess(img)
        img, rectangle_list = find_nummbers(img)
        
        result_images.append(img)
    
    save_all(result_images, path_list)  
    
    
    
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
    # return opening

def get_contour_precedence(contour, cols):
    tolerance_factor = 500
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

def check_edges(w, h, x1, y1, x2, y2):
    x1 = 0 if x1 < 0 else x1
    y1 = 0 if y1 < 0 else y1
    x2 = w if x2 > w else x2
    y2 = h if y2 > h else y2
    return x1, y1, x2, y2
    


def find_nummbers(image):
    thresh = preprocess(image)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda x:get_contour_precedence(x, thresh.shape[1]))
    
    contour = np.zeros(image.shape, np.uint8)
    contour = cv2.drawContours(contour, contours, -1, (255, 255, 255), 8)
    output = gray = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    rectangle_list = []
    spc = 50
    width, height= output.shape[1], output.shape[0]
    cursor_h = int(height*0.25)
    
    for i in range(len(contours)):
        selected = np.zeros(thresh.shape, np.uint8)
        selected = cv2.drawContours(selected, contours[i], -1, 255, 8)
        x,y,w,h = cv2.boundingRect(selected)
        
        
        x1, y1, x2, y2 = check_edges(width, height,
                                      x-spc, y-spc, x+w+spc, y+h+spc)
        
        # x1, y1, x2, y2 = x-spc, y-spc, x+w+spc, y+h+spc
        coords = [(x1, y1), (x2, y2)]
        rectangle_list.append(coords)
        
        
        # cv2.rectangle(output, coords[0], coords[1], (0,0,255), 2)
        # cv2.line(output,(0,cursor_h),(width,cursor_h),(255, 0, 0), 3)
        # cv2.putText(output, str(i),
        # (x1+5, y1+60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        
    rectangle_list = select_and_sort(cursor_h, rectangle_list)
    crapped = crop_numbers(thresh,rectangle_list)
    
    counter = 0
    for rectangle in rectangle_list:
        cv2.rectangle(output, rectangle[0], rectangle[1], (0,0,255), 2)
        cv2.line(output,(0,cursor_h),(width,cursor_h),(255, 0, 0), 3)
        cv2.putText(output, str(counter),
                    (rectangle[0][0]+5, rectangle[0][1]+60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        counter = counter+1
    
    

    cv2.putText(output, "cnt: {}".format(len(contours)),
        (5, output.shape[0]-10), 
        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)

    return crapped, rectangle_list

def crop_numbers(image, rectangle_list):
    # crop_image = image[y:y+h, x:x+w]
    number_list = []
    
    
    for rectangle in rectangle_list:
        x1, y1 = rectangle[0]
        x2, y2 = rectangle[1]
        
        crop_image = image[y1:y2, x1:x2]
        crop_image = resize(crop_image)
        number_list.append(crop_image)
                
    
    im_h_resize = hconcat_resize_min(number_list)
    # vertical = np.vstack((
    #     image,
    #     cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
    #     cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR),
    #     cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR),
    #     contour_image))
    # return number_list
    return im_h_resize

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)



def select_and_sort(cursor, rectangle_list):
    select_list = []
    for rectangle in rectangle_list:
        x1, y1 = rectangle[0]
        x2, y2 = rectangle[1]
        if(y1<cursor and cursor<y2):
            select_list.append(rectangle)
    return sorted(select_list, key=lambda x: x[0][0])
    
    

def deneme(thresh):
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    for i in range(num_labels):
        leftmost_x = stats[i, cv2.CC_STAT_LEFT]
        topmost_y = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        # enclose all detected components in a blue rectangle
        cv2.rectangle(thresh, (leftmost_x, topmost_y), (leftmost_x + width, topmost_y + height), (255, 0, 0), 2)
    

if  __name__ == "__main__":   
   run()
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   