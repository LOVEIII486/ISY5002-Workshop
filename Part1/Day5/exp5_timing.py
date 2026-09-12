# -*- coding: utf-8 -*-
"""
Timing probe for Day5 workshop.
Measures how long ONE epoch takes on CPU with the lecturer's base model,
so the architecture search can be budgeted without running long training.
"""
import os
import time

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 29


def load_raw():
    trDat = np.load(os.path.join(HERE, "kmnist-train-imgs.npz"))["arr_0"]
    trLbl = np.load(os.path.join(HERE, "kmnist-train-labels.npz"))["arr_0"]
    tsDat = np.load(os.path.join(HERE, "kmnist-test-imgs.npz"))["arr_0"]
    tsLbl = np.load(os.path.join(HERE, "kmnist-test-labels.npz"))["arr_0"]

    trDat = trDat.astype("float32") / 255
    tsDat = tsDat.astype("float32") / 255

    trDat = trDat.reshape(trDat.shape[0], trDat.shape[1], trDat.shape[2], 1)
    tsDat = tsDat.reshape(tsDat.shape[0], tsDat.shape[1], tsDat.shape[2], 1)

    return trDat, to_categorical(trLbl), tsDat, to_categorical(tsLbl)


def base_model(num_classes):
    """The lecturer's structure (slide 36-39), total params = 103,898."""
    m = Sequential()
    m.add(Conv2D(20, (5, 5), input_shape=(28, 28, 1), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(40, (5, 5), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.2))
    m.add(Flatten())
    m.add(Dense(128, activation="relu"))
    m.add(Dense(num_classes, activation="softmax"))
    m.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return m


if __name__ == "__main__":
    np.random.seed(SEED)
    trDat, trLbl, tsDat, tsLbl = load_raw()
    print("data ready: train", trDat.shape, "test", tsDat.shape, flush=True)

    model = base_model(trLbl.shape[1])

    # warm up the TF graph so the timed epoch is not paying graph-build cost
    t0 = time.time()
    model.fit(trDat[:1], trLbl[:1], epochs=1, verbose=0)
    print("warmup done in %.1fs" % (time.time() - t0), flush=True)

    t0 = time.time()
    h = model.fit(trDat, trLbl, validation_data=(tsDat, tsLbl),
                  epochs=1, batch_size=128, verbose=1)
    dt = time.time() - t0

    print("ONE FULL EPOCH: %.1fs (train_acc=%.4f val_acc=%.4f)"
          % (dt, h.history["accuracy"][0], h.history["val_accuracy"][0]), flush=True)
    print("=> 10 epochs approx %.1f min, 60 epochs approx %.1f min"
          % (dt * 10 / 60, dt * 60 / 60), flush=True)
