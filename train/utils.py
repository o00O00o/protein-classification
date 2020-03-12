import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


def one_hot_encoder(my_string):
    my_array = np.array(list(my_string))
    categories = np.array([
        'A', 'C', 'E', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T',
        'U', 'V', 'Y', '3', '2', 'K', '1', '5', '7', '0', 'D'
    ])
    label_encoder = LabelEncoder()
    label_encoder.fit(categories)
    integer_encoded = label_encoder.transform(my_array)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = to_categorical(integer_encoded, num_classes=25)
    return onehot_encoded


def normalization_processing(data):
    data_mean = data.mean()
    data_var = data.std()
    data = data - data_mean
    data = data / data_var
    return data
