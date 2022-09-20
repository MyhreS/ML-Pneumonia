import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

def create_model(first_conv_layer_filter, second_conv_layer_filter, third_conv_layer_filter, optimizer, activation,
                 loss_function, learning_rate, IMG_SHAPE):


    # Create model
    model = Sequential([])
    model.add(Input(shape=IMG_SHAPE))
    model.add(Rescaling(1./255))
    model.add(Conv2D(first_conv_layer_filter, kernel_size=(3, 3), activation=activation))
    model._name += "_conv-{}".format(first_conv_layer_filter)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model._name += "_maxpool"
    model.add(Conv2D(second_conv_layer_filter, kernel_size=(3, 3), activation=activation))
    model._name += "_conv-{}".format(second_conv_layer_filter)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model._name += "_maxpool"
    model.add(Conv2D(third_conv_layer_filter, kernel_size=(3, 3), activation=activation))
    model._name += "_conv-{}".format(third_conv_layer_filter)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model._name += "_maxpool"
    model.add(Flatten())
    model.add(Dense(128, activation=activation))
    model._name += "_dense-128"
    model.add(Dropout(0.5))
    model._name += "_dropout"
    model.add(Dense(64, activation=activation))
    model._name += "_dense-64"
    model.add(Dense(1, activation="sigmoid"))

    model._name += "_optimizer-{}_activation-{}_loss-{}_learning_rate-{}".format(optimizer, activation, loss_function, learning_rate)
    # Compile model
    if optimizer == "adam":
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=loss_function,
            metrics=['accuracy']
        )
    elif optimizer == "sgd":
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=loss_function,
            metrics=['accuracy']
        )
    elif optimizer == "rmsprop":
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
            loss=loss_function,
            metrics=['accuracy']
        )
    else:
        model.compile(optimizer=optimizer,
                      loss=loss_function,
                      metrics=['accuracy'])

    return model