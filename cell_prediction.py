import warnings   
warnings.filterwarnings('ignore')
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import flask
from flask import Flask, jsonify, request
import json


app = Flask(__name__)
@app.route('/predict', methods=['GET'])

def predict():
    model=load_model('model/malaria_detector.h5')
    file=input("Enter the Cell Image Path:")
    def Convert(string): 
        list_op= list(string.split(" ")) 
        return list_op
    image_path=Convert(file)
    image_shape = (128,128,3)
    result=[]
    for img in image_path:
        cell= image.load_img(img,target_size=image_shape)
        cell_img = image.img_to_array(cell)
        cell_img = np.expand_dims(cell_img, axis=0)
        out=model.predict(cell_img)
        if out==1:
            output='Uninfected'
        else :
            output='Parasitized'
        result.append(output)    
    
    return print(str(result))

if __name__ == '__main__':
    app.run(debug=True)