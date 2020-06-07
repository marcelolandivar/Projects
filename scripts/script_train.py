from Covid_Classifier.train import ImportNpy, DataPreprocessing

path = r"/home/marcelo-landivar/PycharmProjects/Projects/dataset"


"""
---------------- PIPELINE SECTION 1 ------------------------
"""
img, lab = ImportNpy(dataset_path=path).concatenate_arrays()

"""
---------------- PIPELINE SECTION 2 ------------------------
"""

trainX, testX, trainY, testY = DataPreprocessing(images=img, labels=lab).split_train_test()

print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)

"""
---------------- PIPELINE SECTION 3 ------------------------
"""
m = Models()


"""
---------------- PIPELINE SECTION 4 ------------------------
"""
p=Predict(m)