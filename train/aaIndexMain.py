import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.optimizers import adam
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from models import aaIndexCNN
from generators import aaIndexGenerate

if __name__ == "__main__":

    # 准备数据
    df = pd.read_csv("ProCla/data/AA-index/S/encoded.txt", dtype=np.float32)
    data = df.iloc[:, 1:]
    data = np.array(data)
    label = df.iloc[:, 0]
    label = np.array(label)
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.20, random_state=42)

    # 训练设置
    train_num = len(x_train)
    test_num = len(x_test)
    train_batch = 16
    test_batch = 16

    adam = adam(lr=0.0001)
    model = aaIndexCNN()
    checkPoint = ModelCheckpoint("ProCla/models", monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=2)
    callbacksList = [checkPoint]
    model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=["accuracy"])
    model.summary()

    results = model.fit_generator(aaIndexGenerate(x_train, y_train, train_batch),
                                  steps_per_epoch=train_num / train_batch, epochs=100,
                                  validation_data=aaIndexGenerate(x_test, y_test, test_batch),
                                  validation_steps=test_num / test_batch,
                                  verbose=1, callbacks=callbacksList)

    # 绘制loss和acc曲线
    loss = results.history['loss']
    val_loss = results.history['val_loss']
    acc = results.history['acc']
    val_acc = results.history['val_acc']
    epochs = range(1, len(loss) + 1)

    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('validation accuracy')
    plt.legend()
    plt.show()
