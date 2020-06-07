import os
import cv2
import numpy as np
from numpy import save

class DirTools(object):
    """
    Class Directory tools
    """

    def __init__(self, path):
        self.path = path

    def get_datasets_paths(self):
        """
        It gets normal and covid paths from base dataset
        :return the path to Covid and Normal datasets
        """
        global dir_covid
        global dir_normal

        #Loop through directories, subdirs and files for dir, subdir, file in os.walk(self.path)L

        for dir, subdir, file in os.walk(self.path):

            #Register last folder
            last_folder = os.path.basename(os.path.normpath(dir))

            #Check if last folder is covid
            if last_folder == 'covid':
                dir_covid = dir

            #Check if last folder is normal
            elif last_folder == 'normal':
                dir_normal = dir

            elif last_folder == 'saved':
                dir_saved = dir

        return dir_covid, dir_normal, dir_saved

class ImagePreprocessing(object):
    """
    Preprocess the images for training
    """

        # Setting the dataset paths

    def __init__(self, image_path):
        self.image_path = image_path

    def _save_npy(self, name, data):
        """
        Save a NPY from data
        :param name: name with .npy extension
        :param data: numpy array
        :return: saved npy
        """
        os.makedirs(os.path.join(os.path.dirname(self.image_path), 'saved'), exist_ok=True)
        storage = os.path.join(os.path.dirname(self.image_path), 'saved')

        save(os.path.join(storage, name), data)


    def get_images_to_list(self, save_to_npy=False):
        """
        Gets images an return a list
        :return: A list of images and labels in a numpy array
        """

        images = []
        labels = []
        # Extract the class label from the file path (covid or normal)
        label = os.path.basename(os.path.normpath(self.image_path))


        #Loop through images and prepare them for training
        for image in os.listdir(self.image_path):

            print("Processing image:", image)

            impath = os.path.join(self.image_path, image)

            image = cv2.imread(impath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))

            # Update images and labels lists, respectively
            images.append(image)
            labels.append(label)
            """ 
            Convert the data and labels to NumPy arrays while scaling the pixel
            intensities to the range [0, 1] 
            """

        images = np.array(images) / 255.0
        labels = np.array(labels)

        if save_to_npy:

            self._save_npy(name="array_{}.npy".format(label), data=images)
            self._save_npy(name="labels_{}.npy".format(label), data=labels)


        return images, labels
