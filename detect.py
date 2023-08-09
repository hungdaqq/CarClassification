import numpy as np
import cv2 as cv
import requests
import pickle

from keras.models import load_model

# # Input shape for ANN and also to resize images 
input_shape = (224, 224, 3)

# Loading best weights
model = load_model('./working/last.h5')
class_name = pickle.loads(open('./working/class_name.pickle', "rb").read())
class_name_list = np.array(class_name).tolist()

def class_sort(item):
    return class_name_list.index(item)

urls = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Kia_Optima_%28JF%29_%28cropped%29.jpg/1200px-Kia_Optima_%28JF%29_%28cropped%29.jpg"
]

for i, url in enumerate(urls):    
    # Sending request to the URL
    r = requests.get(url, stream = True).raw
    
    # Reading image, convert it to np array and decode
    image = np.asarray(bytearray(r.read()), dtype="uint8")
    image = cv.imdecode(image, cv.IMREAD_COLOR)
    
    # Resize, scale and reshape image before making predictions
    resized = cv.resize(image, (input_shape[1], input_shape[0]))
    resized = resized.reshape(-1, input_shape[1], input_shape[0], input_shape[2])
    
    # Predict results
    predictions = model.predict(resized)
    predictions = zip(list(class_name), list(predictions[0]))
    predictions = sorted(list(predictions), key = lambda z: z[1], reverse = True)
    final_preds = []

    for index, probability in predictions[:3]:
        final_preds.append(index)
        probability *= probability        
        
    final_preds = sorted(final_preds, key=class_sort)
    print(final_preds[0] + '_' + final_preds[1] + '_' + final_preds[2] + '_' + str(probability))
