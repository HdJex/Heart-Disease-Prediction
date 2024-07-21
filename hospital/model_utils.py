import numpy as np
from django.conf import settings
import tensorflow as tf
from tensorflow import keras
from keras.applications.mobilenet import MobileNet 
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.models import model_from_json
import keras
from keras import backend as K

from .skin_classes import SKIN_CLASSES




def load_model_and_predict(image_path):
    j_file = open(settings.BASE_DIR / 'dataset' / 'modelnew.json', 'r')
    loaded_json_model = j_file.read()
    j_file.close()
    
    model = model_from_json(loaded_json_model)
    model.load_weights(settings.BASE_DIR / 'dataset' / 'modelnew.h5')
    
    img1 = image.load_img(image_path, target_size=(224, 224))
    img1 = np.array(img1)
    img1 = img1.reshape((1, 224, 224, 3))
    img1 = img1 / 255
    
    prediction = model.predict(img1)
    pred = np.argmax(prediction)
    disease = SKIN_CLASSES[pred]
    accuracy = prediction[0][pred]
    
    K.clear_session()
    return disease, accuracy, image_path.split('/')[-1]


