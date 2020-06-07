from Covid_Classifier.etl import DirTools, ImagePreprocessing

path = r'/home/marcelo-landivar/PycharmProjects/Projects/dataset'

covid, normal, _ = DirTools(path=path).get_datasets_paths()

covid_images, covid_labels= ImagePreprocessing(image_path=covid).get_images_to_list(save_to_npy=True)
normal_images, normal_labels = ImagePreprocessing(image_path=normal).get_images_to_list(save_to_npy=True)








