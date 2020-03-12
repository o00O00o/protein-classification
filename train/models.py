from keras.layers import Dense, Input, Activation, Flatten, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import MaxPooling2D, Conv2D
from keras.models import Model


def oneHotCNN():

    inputs = Input((500, 25, 1))

    x = Conv2D(16, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs, x)

    return model


def aaIndexCNN():

    inputs = Input((11151, 1))

    x = Conv1D(8, 32, padding="same")(inputs)
    x = MaxPooling1D(4)(x)
    x = Conv1D(16, 24, padding="same")(x)
    x = MaxPooling1D(4)(x)
    x = Conv1D(32, 12, padding="same")(x)
    x = MaxPooling1D(4)(x)

    x = Conv1D(64, 9, padding="same")(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 6, padding="same")(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(256, 3, padding="same")(x)
    x = MaxPooling1D(2)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs, x)

    return model
