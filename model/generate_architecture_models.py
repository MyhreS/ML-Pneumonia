
def add_conv_layer(a, model_stack):
    if a == 0:
        model_stack.append(["Conv2D", 16])
    elif a == 1:
        model_stack.append(["Conv2D", 32])
    else:
        model_stack.append(["Conv2D", 64])

def add_dense_layer(a, model_stack):
    if a == 0:
        model_stack.append(["Dense", 128])
    elif a == 1:
        model_stack.append(["Dense", 64])
    elif a == 2:
        model_stack.append(["Dense", 32])
    else:
        model_stack.append(["Dense", 16])

def add_between_layer(count, i, model_stack):
    # Add batchnorm or dropout 2 every second time after conv layer. Starting with batchnorm
    if i == 0 and count == 0:
        model_stack.append(["BatchNormalization", ])
    elif i == 0 and count == 1:
        model_stack.append(["Dropout", 0.2])
    # Add dropout or batchnorm 2 every second time after conv layer. Starting with dropout
    elif i == 1 and count == 0:
        model_stack.append(["Dropout", 0.2])
    elif i == 1 and count == 1:
        model_stack.append(["BatchNormalization", ])
    # Add dropout after every conv layer.
    elif i == 2:
        model_stack.append(["Dropout", 0.2])
    # Add batchnorm after every conv layer.
    elif i == 3:
        model_stack.append(["BatchNormalization", ])
    # Add dropout every second layer after conv layer.
    elif i == 4 and count == 0:
        model_stack.append(["Dropout", 0.2])
    elif i == 4 and count == 1:
        pass
    # Add batchnorm every second layer after conv layer.
    elif i == 5 and count == 0:
        model_stack.append(["BatchNormalization", ])
    elif i == 5 and count == 1:
        pass








def create_model(numb_conv_layers, numb_dense_layers, i):
    model_stack = []

    # Add conv layer
    count = 0
    for a in range(numb_conv_layers-1):
        # Add conv layer
        add_conv_layer(a, model_stack)
        model_stack.append(["MaxPooling2D", ])
        # Add between layer
        add_between_layer(count, i, model_stack)
        # Increase count
        count += 1
        if count == 2:
            count = 0

    # Add last conv
    model_stack.append(["Conv2D", 64])
    # Add flatten
    model_stack.append(["Flatten", ])

    # Add dense
    for a in range(numb_dense_layers):
        add_dense_layer(a, model_stack)
    return model_stack





def generate_models(conv_layers, dense_layers):
    models = []
    for conv_layer in conv_layers:
        for dense_layer in dense_layers:
            for i in range(6):
                model = create_model(conv_layer, dense_layer, i)
                models.append(model)
    for model in models:
        print(model)
    return models












