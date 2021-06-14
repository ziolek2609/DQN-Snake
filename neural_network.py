from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class NeuralNetwork():

    def __init__(self, input, output, learningRate):
        self.input = input
        self.output = output
        self.learningRate = learningRate

        self.model = Sequential()
        self.model.add(Dense(units=18,
                             activation='relu',
                             input_shape=(self.input, )))
        self.model.add(Dense(units=18,
                             activation='relu'))
        self.model.add(Dense(units=self.output,
                             activation='softmax'))
        self.model.compile(optimizer=Adam(lr=self.learningRate),
                           loss='mae')
