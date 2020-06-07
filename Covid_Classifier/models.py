import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, LambdaCallback


class Models(object):

    def __init__(self, num_neurons_first, dropout, init_lr, epochs):
        self.num_neurons_first = num_neurons_first
        self.dropout = dropout
        self.init_lr = init_lr
        self.epochs = epochs

    def vgg16(self):
        """
        Load the VGG16 network, ensuring the head (output) FC layer sets are left off
        """
        baseModel = VGG16(weights="imagenet", include_top=False,
                          input_tensor=Input(shape=(224, 224, 3)))

        """
        # Construct the head of the model that will be placed 
        on top of the base model, i.e. after the convolutional layers
        """
        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(self.num_neurons_first, activation="relu")(headModel)
        headModel = Dropout(self.dropout)(headModel)
        headModel = Dense(2, activation="softmax")(headModel)

        """
        Generat a new neural network by placing the head FC model 
        on top of the base model (this will become the actual model we will train)
        """
        model = Model(inputs=baseModel.input, outputs=headModel)

        """
        Loop over all layers in the base model and freeze them 
        so they will *not* be updated during the first training process
        """
        for layer in baseModel.layers:
            layer.trainable = False

        opt = Adam(lr=self.init_lr, decay=self.init_lr / self.epochs)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

        return model