from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical


class MyRNN(keras.Model):
    def __init__(self, n_class):
        super(MyRNN, self).__init__(name="my_rnn")
        self.n_class = n_class
        self.lstm = LSTM(128)
        self.dense = Dense(n_class, activation="softmax")

    def call(self, x, training=False):
        x = self.lstm(x)
        x = self.dense(x)
        return x


def preprocess(data, is_label=False):
    if is_label:
        return to_categorical(data)
    else:
        data = data.astype("float32") / 255
        return data.reshape((-1, 28, 28))


if __name__ == "__main__":
    n_classes = 10
    batch_size = 128
    n_epochs = 5

    # # create_model
    model = MyRNN(n_classes)
    model.compile(
        loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]
    )

    # load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # train
    x_train = preprocess(x_train)
    x_test = preprocess(x_test)
    y_train = preprocess(y_train, is_label=True)
    y_test = preprocess(y_test, is_label=True)

    # print(x_train.shape)

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=n_epochs,
        shuffle=True,
    )

    # test
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"{loss=}, {accuracy=}")
