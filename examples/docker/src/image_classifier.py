import io
import requests
import logging
import os

from PIL import Image
from pathlib import Path
from fastai.vision import ImageDataBunch, get_transforms, models, cnn_learner, accuracy, load_learner, open_image
from htx.base_model import SingleClassImageClassifier
from htx.utils import download

from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.applications.nasnet import NASNetMobile
from keras.layers import Dense
from keras.optimizers import SGD
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator


logger = logging.getLogger(__name__)


class FastaiImageClassifier(SingleClassImageClassifier):

    def load(self, serialized_train_output):
        # self._model = load_learner(serialized_train_output['model_path'])
        self._model = load_model(serialized_train_output['model_path'])
        self._image_dir = serialized_train_output['image_dir']

    @classmethod
    def _get_image_from_url(self, url):
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with io.BytesIO(r.content) as f:
            return Image.open(f).convert('RGB')
    

    def predict(self, tasks, **kwargs):
        pred_labels, pred_scores = [], []
        list_of_labels = ["Back","Discard","Front","Left","Right"]
        for task in tasks:
            image_file = download(task['input'][0], self._image_dir)
            image = load_img(image_file, target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            probs = self._model.predict(image)[0]
            label_idx = np.argmax(probs)
            label = list_of_labels[label_idx]
            score = probs[label_idx]
            pred_labels.append(label)
            pred_scores.append(score.item())
        return self.make_results(tasks, pred_labels, pred_scores)
        

    # def predict(self, tasks, **kwargs):
    #     pred_labels, pred_scores = [], []
    #     for task in tasks:
    #         image_file = download(task['input'][0], self._image_dir)
    #         _, label_idx, probs = self._model.predict(open_image(image_file))
    #         label = self._model.data.classes[label_idx]
    #         score = probs[label_idx]
    #         pred_labels.append(label)
    #         pred_scores.append(score.item())
    #     return self.make_results(tasks, pred_labels, pred_scores)


num_class = 5
def creat_compiled_model():
    model = Sequential()
    model.add(NASNetMobile(input_shape=(224, 224, 3), include_top=False,
                           weights='imagenet', pooling='avg'))
    model.add(Dense(num_class,activation='softmax'))
    model.layers[0].trainable = True
    model.layers[1].trainable = True
    sgd = SGD(learning_rate=0.01, momentum=0.01, nesterov=False)
    model.compile(optimizer=sgd, loss='hinge',  metrics=['accuracy'])
    return model 

model = creat_compiled_model()


def train_script(input_data, output_dir, image_dir, batch_size=4, num_iter=10, **kwargs):
    """
    This script provides FastAI-compatible training for the input labeled images
    :param image_dir: directory with images
    :param filenames: image filenames
    :param labels: image labels
    :param output_dir: output directory where results will be exported
    :return: fastai.basic_train.Learner object
    """

    filenames, labels = [], []
    for item in input_data:
        if item['output'] is None:
            continue
        image_url = item['input'][0]
        label = item['output'][0]
        label = str(label) 
        image_path = download(image_url, os.path.join(image_dir, label))
        # image_path = download(image_url, image_dir + "/" + label)
        # Les deux prochains lignes peuvent être supprime
        filenames.append(image_path)
        labels.append(label)
    # donc filenames contient la liste des path vers les images et labels leur label
    # maintenant on va creer les dossier tq chaque dossier 
    # a ce stade le dossier image_dir contient 5 sous dossier qui corespond aux labels
    train_datagen = ImageDataGenerator(rescale=1./255,preprocessing_function=preprocess_input)
    train_generator = train_datagen.flow_from_directory(image_dir, batch_size=batch_size)
    model.fit_generator(train_generator, steps_per_epoch=3, epochs=num_iter)
    model.save_weights(output_dir + "/keras_model.h5")
    return {'model_path': output_dir, 'image_dir': image_dir}



# def train_script(input_data, output_dir, image_dir, batch_size=4, num_iter=10, **kwargs):
#     """
#     This script provides FastAI-compatible training for the input labeled images
#     :param image_dir: directory with images
#     :param filenames: image filenames
#     :param labels: image labels
#     :param output_dir: output directory where results will be exported
#     :return: fastai.basic_train.Learner object
#     """

#     filenames, labels = [], []
#     for item in input_data:
#         if item['output'] is None:
#             continue
#         image_url = item['input'][0]
#         image_path = download(image_url, image_dir)
#         filenames.append(image_path)
#         labels.append(item['output'][0])

#     tfms = get_transforms()
#     data = ImageDataBunch.from_lists(
#         Path(image_dir),
#         filenames,
#         labels=labels,
#         ds_tfms=tfms,
#         size=224,
#         bs=batch_size
#     )
#     learn = cnn_learner(data, models.resnet18, metrics=accuracy, path=output_dir)
#     learn.fit_one_cycle(num_iter)
#     learn.export()
#     return {'model_path': output_dir, 'image_dir': image_dir}
