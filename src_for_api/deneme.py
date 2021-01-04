from PIL import Image
import cv2
from datetime import date
from datetime import datetime
from run_model import make_prediction



def time_stamp():
    today = date.today()
    d1 = today.strftime("%d/%m/%Y")

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    print(d1,current_time)
    return now


if __name__ == "__main__": 
    start = time_stamp()

    image = Image.open("..\\data\\model_input_images\\e7.jpg")
    print(type(image))

    cropped_image = make_prediction(image)
    # show_image = cv2.resize(cropped_image,(1024,768))
    cv2.imshow("dvf",cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    end = time_stamp()
    print("prediction duration: ", end - start)