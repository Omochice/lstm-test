import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM


def _make_noised_sin():
    STEPS_PER_CYCLE = 80
    N_CYCLES = 50
    df = pd.DataFrame(np.arange(STEPS_PER_CYCLE * N_CYCLES + 1), columns=["t"])
    # df["sin_t"] = df.t.apply(lambda x: math.sin(
    #     x * (2*math.pi / STEPS_PER_CYCLE) + random.uniform(-0.05, 0.05)))
    df["sin_t"] = df.t.apply(lambda x: math.sin(
        x * (2 * math.pi / STEPS_PER_CYCLE) + random.uniform(-0.05, +0.05)))
    # print(df.head(5))
    return df


def _load_data(data, n_prev=30):
    docX, docY = [], []

    for i in range(len(data) - n_prev):
        # docX.append(data.iloc[i:i+n_prev].values)
        # docY.append(data.iloc[i+n_prev].values)
        docX.append(data.iloc[i:i + n_prev].values)
        docY.append(data.iloc[i + n_prev])
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY


def train_test_split(df, test_size=0.1, n_prev=30):
    ntrn = int(round(len(df) * (1 - test_size)))
    x_train, y_train = _load_data(data=df.iloc[0:ntrn], n_prev=n_prev)
    x_test, y_test = _load_data(data=df.iloc[ntrn:], n_prev=n_prev)

    return (x_train, y_train), (x_test, y_test)


def ml():
    # param
    in_out_neurons = 1
    hidden_neurons = 300
    length_of_sequences = 30

    df = _make_noised_sin()

    (x_train, y_train), (x_test, y_test) = train_test_split(df[["sin_t"]])

    model = Sequential()
    model.add(LSTM(hidden_neurons, batch_input_shape=(
        None, length_of_sequences, in_out_neurons), return_sequences=False))
    model.add(Dense(in_out_neurons))
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer="adam")

    print(model.summary())

    print(f"{len(x_train)=}, {len(x_train[1])=}")
    model.fit(x_train, y_train, batch_size=600, epochs=15, validation_split=0.05)

    # predict
    predicted = model.predict(x_test)

    dataf = pd.DataFrame(predicted[:200])
    dataf.columns = ["predict"]
    dataf["input"] = y_test[:200]
    dataf.plot()
    plt.savefig("hoge.png")


if __name__ == "__main__":
    ml()
