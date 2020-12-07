# Python program to explain cv2.rectangle() method  
   
# importing cv2  
# import cv2  
# import  numpy as np
   
# # path  
# image = np.zeros((500,500, 3), dtype=np.uint8)
   
# # Window name in which image is displayed 
# window_name = 'Image'
  
# # Start coordinate, here (5, 5) 
# # represents the top left corner of rectangle 
# start_point = [[5 5]] 
  
# # Ending coordinate, here (220, 220) 
# # represents the bottom right corner of rectangle 
# end_point =  [[220 220]]

# print(type(start_point))
  
# # Blue color in BGR 
# color = (0, 0, 255) 
  
# # Line thickness of 2 px 
# thickness = 2
  
# # Using cv2.rectangle() method 
# # Draw a rectangle with blue line borders of thickness of 2 px 
# image = cv2.rectangle(image, [[5 5]], [[220 220]], color, thickness) 
  
# # Displaying the image  
# cv2.imshow(window_name, image) 
# cv2.waitKey(0) 
from datetime import date
from datetime import datetime



def time_stamp():
    today = date.today()
    d1 = today.strftime("%d/%m/%Y")

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    print(d1,current_time)
    return now


start = time_stamp()
input()
end = time_stamp()
print("training duration: ", end - start)