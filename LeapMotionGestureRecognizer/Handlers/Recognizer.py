from tensorflow.keras.models import load_model
import operator
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Conv1D, AveragePooling1D, Dropout, Flatten, Lambda, Dense, \
    concatenate, LSTM
from tensorflow.python.keras.layers import LeakyReLU, MaxPooling1D
from tensorflow.python.keras.regularizers import L1L2


class Recognizer:
    classes = 3
    sequence_length = 200
    features = 52
    model: Model

    def __init__(self):
        # self.set_model()
        # self.set_model_weights()

        self.load_model()

    def load_model(self):
        self.model = keras.models.load_model("ML_Models/my_model")

    def set_model_weights(self):
        self.model.load_weights("ML_Models/gesture_recognizer_200_time_steps.hdf5")

    def set_model(self):
        l1 = 1e-2
        l2 = 2e-2
        self.model = Sequential()
        self.model.add(Input(shape=(self.sequence_length, self.features)))
        self.model.add(Conv1D(filters=64, kernel_size=3, padding='same',
                              bias_regularizer=L1L2(l1=l1, l2=l2)))
        self.model.add(LeakyReLU(alpha=0.01))
        self.model.add(MaxPooling1D(pool_size=8))
        self.model.add(Dropout(0.2))
        self.model.add(Conv1D(filters=64, kernel_size=3, padding='same',
                              bias_regularizer=L1L2(l1=l1, l2=l2)))
        self.model.add(LeakyReLU(alpha=0.01))
        self.model.add(MaxPooling1D(pool_size=8))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(3, return_sequences=True, bias_regularizer=L1L2(l1=l1, l2=l2)))
        self.model.add(LSTM(3, bias_regularizer=L1L2(l1=1e-1, l2=1e-1)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(40, bias_regularizer=L1L2(l1=l1, l2=l2)))
        self.model.add(LeakyReLU(alpha=0.01))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.classes, activation='sigmoid'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], run_eagerly=True)
        self.model.build()
        print(self.model.summary())

    def infer(self, x):
        return self.model.predict(np.array([x]))[0].tolist()

    def get_classes(self):
        return ['wiggle', 'grab', 'hand_flick']

    def get_class(self, prediction):
        index, value = max(enumerate(prediction), key=operator.itemgetter(1))
        return self.get_classes()[index], value
