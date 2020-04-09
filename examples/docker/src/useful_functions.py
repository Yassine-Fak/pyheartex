import io
import os
import hashlib
import requests
from PIL import Image
from glob import glob
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def download_image_from_url(url, output_dir, filename=None):
    # This function allows us to download a picture from a given url
    # It returns the file path of the downloaded image
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if filename is None:
        filename = hashlib.md5(url.encode()).hexdigest()
    filepath = os.path.join(output_dir, filename)
    if os.path.exists(filepath):
        im = Image.open(filepath)
        os.remove(filepath)
        filepath = os.path.join(output_dir, filename + ".jpg")
        im.save(filepath)
        return filepath
    r = requests.get(url)
    r.raise_for_status()
    with io.open(filepath, mode='wb') as fout:
        fout.write(r.content)
    im = Image.open(filepath)
    os.remove(filepath)
    filepath = os.path.join(output_dir, filename + ".jpg")
    im.save(filepath)
    return filepath

num_class = 5
# TODO cette fonction doit prendre en consideration les differents classes 
# TODO ajouter unn attribut dans l'objet model
def creat_compiled_model():
    model = Sequential()
    model.add(NASNetMobile(input_shape=(224, 224, 3), include_top=False,
                           weights='imagenet', pooling='avg'))
    model.add(Dense(num_class, activation='softmax'))
    model.layers[0].trainable = True
    model.layers[1].trainable = True
    sgd = SGD(learning_rate=0.01, momentum=0.01, nesterov=False)
    model.compile(optimizer=sgd, loss='hinge', metrics=['accuracy'])
    return model


model = creat_compiled_model()

