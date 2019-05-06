from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential


class CNN(Sequential):
    def __init__(self, classes_num=7, input_shape=(48, 48, 1)):
        super().__init__()

        self.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        self.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.25))

        self.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.25))

        self.add(Flatten())
        self.add(Dense(1024, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(classes_num, activation='softmax'))