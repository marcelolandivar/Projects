import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class Predict(object):


    def __init__(self, model, testX, testY, batch_size):
        self.model = model
        self.testX = testX
        self.testY = testY
        self.batch_size = batch_size

    def predict_covid(self):
        pred_ids = self.model(self.testX, batch_size=self.batch_size)
        pred_ids = np.argmax(pred_ids, axis=1)

        print(classification_report(self.testY, np.argmax(pred_ids, axis=1) ))