import numpy as np
from utils import one_hot_encoder, normalization_processing
from keras.utils import to_categorical


def oneHotGenerate(dataSet, labSet, batchSize):

    dataSet = list(dataSet)
    labSet = list(labSet)

    data_list = np.zeros((batchSize, 500, 25, 1), dtype=np.float32)
    label_list = np.zeros((batchSize))

    setNum = len(dataSet)
    batchFlag = 0
    setFlag = 0

    while True:

        data = dataSet[setFlag]
        label = labSet[setFlag]

        data = one_hot_encoder(data)
        data = normalization_processing(data)

        data_list[batchFlag, :, :, 0] = data
        label_list[batchFlag] = label

        batchFlag += 1
        setFlag += 1

        if setFlag >= setNum:
            setFlag = 0

        if batchFlag >= batchSize:

            oneHotLab = to_categorical(label_list, num_classes=2)
            yield [data_list], [oneHotLab]

            batchFlag = 0
            data_list = np.zeros((batchSize, 500, 25, 1), dtype=np.float32)
            lab_list = np.zeros((batchSize))


def aaIndexGenerate(dataSet, labSet, batchSize):

    dataSet = list(dataSet)
    labSet = list(labSet)

    data_list = np.zeros((batchSize, 11151, 1), dtype=np.float32)
    label_list = np.zeros((batchSize))

    setNum = len(dataSet)
    batchFlag = 0
    setFlag = 0

    while True:

        data = dataSet[setFlag]
        label = labSet[setFlag]

        data = normalization_processing(data)

        data_list[batchFlag, :, 0] = data
        label_list[batchFlag] = label

        batchFlag += 1
        setFlag += 1

        if setFlag >= setNum:
            setFlag = 0

        if batchFlag >= batchSize:

            oneHotLab = to_categorical(label_list, num_classes=2)
            yield [data_list], [oneHotLab]

            batchFlag = 0
            data_list = np.zeros((batchSize, 11151, 1), dtype=np.float32)
            lab_list = np.zeros((batchSize))
