from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential


def create_model(generated_model, input_shape, classes, learning_rate):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Rescaling(1. / 255))
    for layer in generated_model:
        if layer[0] == "Conv2D":
            model.add(Conv2D(layer[1], (3, 3), activation='relu'))
            model._name += "_{}-{}".format(layer[0], layer[1])
        elif layer[0] == "MaxPooling2D":
            model.add(MaxPooling2D((2, 2)))
            model._name += "_{}".format(layer[0])
        elif layer[0] == "BatchNormalization":
            model.add(BatchNormalization())
            model._name += "_{}".format(layer[0])
        elif layer[0] == "Dropout":
            model.add(Dropout(layer[1]))
            model._name += "_{}-{}".format(layer[0], layer[1])
        elif layer[0] == "Flatten":
            model.add(Flatten())
        elif layer[0] == "Dense":
            model.add(Dense(layer[1], activation='relu'))
            model._name += "_{}-{}".format(layer[0], layer[1])
    model.add(Dense(len(classes)-1, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    return model