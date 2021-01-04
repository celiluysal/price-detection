# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 19:53:10 2020

@author: a-t-g
"""

  
from PIL import Image
from PIL import ImageFilter
import requests
from io import BytesIO
from flask import Flask, request, jsonify
import pytesseract
import cv2, copy
import numpy as np
from run_model import make_prediction

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

app = Flask(__name__)
_VERSION = 1  # API version


# url = "https://www.sunpaitag.com/upfile/products/9/Electronic_price_tag_italy.png"
url = "https://linkpicture.com/q/e7_1.jpg"


def process_image(url):
    image = _get_image(url)

    # image_cv2 = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)

    image = make_prediction(image)
    cv2.imshow("image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # w,h = image.size
    text = "asd"
    # text = str(w) + str(h)
    #text = pytesseract.image_to_string(image)
    # text = pytesseract.image_to_string(image, lang="tur", config='--psm 6 -c tessedit_char_whitelist=0123456789')

    custom_config = r'-l tur --oem 1 --psm 8' #+ 'outbase digits'
    out_below = pytesseract.image_to_string(img, config=custom_config)

    print(text)
    return text

def _get_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img
    
@app.route('/v{}/ocr'.format(_VERSION), methods=["POST"])
def ocr():
    try:
        url = request.args.get('image_url')
        output = process_image(url)
        return jsonify({"output": output})
    except:
        return jsonify(
            {"error"}
        )
    
@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__': 
    # app.run(debug=True)
    process_image(url)