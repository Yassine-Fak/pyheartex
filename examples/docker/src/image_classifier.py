import io
import requests
import logging
import os
from PIL import Image
from htx.base_model import SingleClassImageClassifier
# from htx.utils import download
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from settings import LOG_LEVEL
import logging
from logging.handlers import RotatingFileHandler
from useful_functions import download_image_from_url
from useful_functions import model
from useful_functions import num_class


logger = logging.getLogger(__name__)
logging_levels = {
    'CRITICAL': logging.CRITICAL,
    'ERROR': logging.CRITICAL,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG
}
logger.setLevel(logging_levels[LOG_LEVEL])

# logging.basicConfig(level=log_levels[LOG_LEVEL])

# création d'un formateur qui va ajouter le temps, le niveau
# de chaque message quand on écrira un message dans le log
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
# création d'un handler qui va rediriger une écriture du log vers
# un fichier en mode 'append', avec 1 backup et une taille max de 1Mo
file_handler = RotatingFileHandler('/data/activity.log', 'a', 1000000, 1)
# on lui met le niveau sur DEBUG, on lui dit qu'il doit utiliser le formateur
# créé précédement et on ajoute ce handler au logger
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# création d'un second handler qui va rediriger chaque écriture de log
# sur la console
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)


class FastaiImageClassifier(SingleClassImageClassifier):

    def load(self, serialized_train_output):
        # self._model = load_learner(serialized_train_output['model_path'])
        # path = os.path.join(serialized_train_output['model_path'], "trained_model.h5")
        logger.debug(f"Loading model from {serialized_train_output['model_path']}")
        self._model = load_model(serialized_train_output['model_path'])
        logger.debug("Model successfully loaded")
        logger.debug(f"Loading images from {serialized_train_output['image_dir']}")
        self._image_dir = serialized_train_output['image_dir']

    @classmethod
    def _get_image_from_url(self, url):
        logger.debug(f"Trying to get image from url '{url}'")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        logger.debug(f"Successfully get image from url '{url}'")
        with io.BytesIO(r.content) as f:
            return Image.open(f).convert('RGB')
    
    def predict(self, tasks, **kwargs):
        pred_labels, pred_scores = [], []
        list_of_labels = ["Back", "Discard", "Front", "Left", "Right"]
        for task in tasks:
            logger.debug(f"Trying to download the image given in the task '{task}'")
            image_file = download_image_from_url(task['input'][0], "/tmp")
            logger.debug(f"Successfully downloaded image from the given task '{task}'")
            logger.debug(f"The image file is : {image_file} ")
            image = load_img(image_file, target_size=(224, 224))
            logger.debug(f"Successfully loaded the image, its content is : {image}")
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            logger.debug(f"making prediction for the image given in the task '{task}'")
            # cette ligne a verifier le self?
            # probs = self._model.predict(image)[0]
            x = int(self._model.predict_classes(image)[0])
            logger.debug(f"successfully predicting the label of the image given in the task '{task}'")
            # label_idx = np.argmax(probs)
            # label = list_of_labels[label_idx]
            # score = probs[label_idx]
            label = list_of_labels[x]
            score = 0.99
            pred_labels.append(label)
            pred_scores.append(score)
            # pred_scores.append(score.item())
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
        logger.debug("Downloading the image to train the model ")
        image_path = download_image_from_url(image_url, os.path.join(image_dir, label))
        # Les deux prochains lignes peuvent être supprime
        logger.debug("Successfully downloaded the image to train the model ")
        filenames.append(image_path)
        labels.append(label)
    logger.debug(f"The labels are: {labels}")
    # here we raise an error if len(labls) != num_of_classes => Raise an Error: Not enough images to rain the model
    number_of_folders = 0
    number_of_files = 0
    for _, dirnames, filnames in os.walk(image_dir):
        number_of_folders += len(dirnames)
        number_of_files += len(filnames)
    logger.debug(f"The number of images found is :  {number_of_files}")
    logger.debug(f"The image directory is: {image_dir}")
    if number_of_folders != num_class:
        logger.error("The number of sub directories is differents from the number of classes")
        raise FileNotFoundError ("The number of sub directories is differents from the number of classes" )
    # donc filenames contient la liste des path vers les images et labels leur label
    # maintenant on va creer les dossier tq chaque dossier 
    # a ce stade le dossier image_dir contient 5 sous dossier qui corespond aux labels
    logger.debug("creating the image data generator object ")
    train_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)
    logger.debug(f"content of directory {os.listdir(image_dir)}")
    # train_generator = train_datagen.flow_from_directory(image_dir, batch_size=batch_size)
    train_generator = train_datagen.flow_from_directory("/data/images", batch_size=batch_size)
    logger.debug("Fitting the model using the generator  ")
    model.fit(train_generator, epochs=num_iter, steps_per_epoch=3)
    logger.debug("Successfully fitting the model  ")
    path = os.path.join(output_dir, "trained_model")
    logger.debug("Saving the model  ")
    sgd = SGD(learning_rate=0.01, momentum=0.01, nesterov=False)
    model.compile(optimizer=sgd, loss='hinge', metrics=['accuracy'])
    model.save(path, save_format='h5')
    logger.debug(f"Model saved in  '{path}'")
    return {'model_path': path, 'image_dir': image_dir}
    # return {'model_path': output_dir, 'image_dir': image_dir}