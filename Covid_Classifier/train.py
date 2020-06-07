import numpy as np
import os
from Covid_Classifier.etl import DirTools
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class ImportNpy(object):
    """
    This class imports saved NPYs from ETL module
    """

    def __init__(self, dataset_path):
        """
        Requests full path to dataset
        :param dataset_path:
        """
        self.dataset_path = dataset_path

    def _search_npy_folder(self):
        """
        Internal function to search for paths in dataset
        :return:
        """

        covid_path, \
        normal_path, \
        npy_path = DirTools(path=self.dataset_path).get_datasets_paths()

        return npy_path

    def _import_npy(self):
            """
            Imports NPYs from ETL module
            :return:
            """

            npy_path = self._search_npy_folder()
            images_covid = np.load(os.path.join(npy_path, "array_covid.npy"))
            images_normal = np.load(os.path.join(npy_path, "array_normal.npy"))
            labels_covid = np.load(os.path.join(npy_path, "labels_covid.npy"))
            labels_normal = np.load(os.path.join(npy_path, "labels_normal.npy"))

            return images_covid, images_normal, labels_covid, labels_normal

    def concatenate_arrays(self):
            images_covid, images_normal, \
            labels_covid, labels_normal = self._import_npy()

            images = np.concatenate((images_covid, images_normal), axis=0)
            labels = np.concatenate((labels_covid, labels_normal), axis=0)

            return images, labels

class DataPreprocessing(object):

    def __init__(self, images, labels):
            self.images = images
            self.labels = labels

    def _encode_one_hot(self):
            # Use one-hot encoding on the labels
            lb = LabelBinarizer()
            labels = lb.fit_transform(self.labels)
            labels = to_categorical(labels)

            return labels

    def split_train_test(self):
            labels = self._encode_one_hot()
            (trainX, testX, trainY, testY) = train_test_split(self.images, labels,
                                                              test_size=0.20, stratify=labels, random_state=42)

            return trainX, testX, trainY, testY





