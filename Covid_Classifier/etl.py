import cv2
import glob
import os

class DirTools(object):
   """
   Class for Directory tools
   """
    def __init__(self, path):
        self.path = path

    def get_images_in_path(self):


# Setting the dataset paths
dataset = 'dataset'

""" Look over the images in our dataset 
and then initialize our data and labels list
"""

imagePaths = list(paths.list_images(dataset))
data = []
labels = []

# Loop over the image paths
for imagePath in imagePaths:
    # Extract the class label from the file path (covid19 or no_covid19)
    label = imagePath.split(os.path.sep)[-2]
    """ 
    Load the image, swap color channels, and resize it to be a fixed
    224x224 pixels while ignoring aspect ratio . This will make the images
    equal in size so that they are ready for our CNN
    """
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    # Update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

""" 
Convert the data and labels to NumPy arrays while scaling the pixel
intensities to the range [0, 1] 
"""
data = np.array(data) / 255.0
labels = np.array(labels)
