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
import cv2
import numpy as np
from run_model import make_prediction

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

app = Flask(__name__)
_VERSION = 1  # API version


url = "https://www.sunpaitag.com/upfile/products/9/Electronic_price_tag_italy.png"


def process_image(url):
    image = _get_image(url)
    # image_cv2 = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
    # cv2.imshow("image",image_cv2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    image = make_prediction(image)
    #text = pytesseract.image_to_string(image)
    text = pytesseract.image_to_string(image, lang="tur", config='--psm 6 -c tessedit_char_whitelist=0123456789')
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
    app.run()
    # process_image(url)