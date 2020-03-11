import pandas as pd
from sklearn.model_selection import train_test_split
from keras.optimizers import adam
from keras.losses import categorical_crossentropy
import matplotlib.pyplot as plt
from model import CNN
from generator import generate

if __name__ == "__main__":

    # 准备数据
    df = pd.read_csv("ProCla/data/fixlen_data.csv")
    data = df.iloc[:, 1]
    label = df.iloc[:, 2]
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.20, random_state=42)

    # 训练设置
    train_num = len(x_train)
    test_num = len(x_test)
    train_batch = 16
    test_batch = 16

    adam = adam(lr=0.0001)
    model = CNN()
    model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=["accuracy"])
    model.summary()

    results = model.fit_generator(generate(x_train, y_train, train_batch),
                                  steps_per_epoch=train_num / train_batch, epochs=100,
                                  validation_data=generate(x_test, y_test, test_batch),
                                  validation_steps=test_num / test_batch, verbose=1)

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
