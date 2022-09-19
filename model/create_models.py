
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

def create_model(input_shape, num_classes, learning_rate):
    model = Sequential([
        Input(shape=input_shape),
        Rescaling(1. / 255),
        Conv2D(16, 3, activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes-1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_models(input_shape, class_names):
    learning_rates = [0.00001, 0.0001, 0.001]
    num_classes = len(class_names)

    models = []
    for lr in learning_rates:
        models.append(create_model(input_shape, num_classes, lr))
    return models