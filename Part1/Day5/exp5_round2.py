# -*- coding: utf-8 -*-
"""
Day5 workshop - round 2.

Round 1 (15 epochs) put 9 configs inside a 1.0pp band, with the top few
separated by only 0.1-0.4pp - too narrow to trust on a single short run.
Round 2 re-runs the leaders at a longer horizon (30 epochs) to see whether
the ranking holds up, and tests whether the top two changes compose.
"""
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 29
EPOCHS = 30
BATCH = 128


def load_data():
    trDat = np.load(os.path.join(HERE, "kmnist-train-imgs.npz"))["arr_0"]
    trLbl = np.load(os.path.join(HERE, "kmnist-train-labels.npz"))["arr_0"]
    tsDat = np.load(os.path.join(HERE, "kmnist-test-imgs.npz"))["arr_0"]
    tsLbl = np.load(os.path.join(HERE, "kmnist-test-labels.npz"))["arr_0"]
    trDat = (trDat.astype("float32") / 255).reshape(-1, 28, 28, 1)
    tsDat = (tsDat.astype("float32") / 255).reshape(-1, 28, 28, 1)
    return trDat, to_categorical(trLbl), tsDat, to_categorical(tsLbl)


def _head(m, nc, dense=128, dropout=0.2):
    m.add(Dropout(dropout))
    m.add(Flatten())
    m.add(Dense(dense, activation="relu"))
    m.add(Dense(nc, activation="softmax"))
    return m


def cfg_base(nc):
    m = Sequential()
    m.add(Conv2D(20, (5, 5), input_shape=(28, 28, 1), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(40, (5, 5), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    return _head(m, nc, 128, 0.2)


def cfg_drop05(nc):
    m = Sequential()
    m.add(Conv2D(20, (5, 5), input_shape=(28, 28, 1), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(40, (5, 5), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    return _head(m, nc, 128, 0.5)


def cfg_same(nc):
    m = Sequential()
    m.add(Conv2D(20, (5, 5), padding="same", input_shape=(28, 28, 1), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(40, (5, 5), padding="same", activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    return _head(m, nc, 128, 0.2)


def cfg_drop05_same(nc):
    """combo of round-1 leaders: dropout 0.5 + padding same"""
    m = Sequential()
    m.add(Conv2D(20, (5, 5), padding="same", input_shape=(28, 28, 1), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(40, (5, 5), padding="same", activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    return _head(m, nc, 128, 0.5)


def cfg_drop05_same_d256(nc):
    """combo of three: dropout 0.5 + padding same + hidden 256"""
    m = Sequential()
    m.add(Conv2D(20, (5, 5), padding="same", input_shape=(28, 28, 1), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(40, (5, 5), padding="same", activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    return _head(m, nc, 256, 0.5)


CONFIGS = [
    ("base            (round-1 reference)", cfg_base),
    ("drop0.5         (round-1 #1)", cfg_drop05),
    ("same            (round-1 #2)", cfg_same),
    ("drop0.5+same    (combo of top 2)", cfg_drop05_same),
    ("drop0.5+same+d256 (top 3 combo)", cfg_drop05_same_d256),
]


def main():
    trDat, trLbl, tsDat, tsLbl = load_data()
    nc = trLbl.shape[1]
    print("ROUND 2: %d epochs each, full 60k train / 10k test" % EPOCHS, flush=True)
    print("%-36s %10s %10s %8s" % ("config", "best_val", "final_val", "secs"))
    print("-" * 70, flush=True)

    results = []
    for name, builder in CONFIGS:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        np.random.seed(SEED)

        model = builder(nc)
        model.compile(loss="categorical_crossentropy", optimizer="adam",
                      metrics=["accuracy"])
        t0 = time.time()
        h = model.fit(trDat, trLbl, validation_data=(tsDat, tsLbl),
                      epochs=EPOCHS, batch_size=BATCH, verbose=0)
        dt = time.time() - t0
        vals = h.history["val_accuracy"]
        results.append((name, max(vals), vals[-1], dt))
        print("%-36s %9.4f%% %9.4f%% %7.0fs"
              % (name, max(vals) * 100, vals[-1] * 100, dt), flush=True)

    print("-" * 70, flush=True)
    print("ROUND 2 RANKING (best val_accuracy, %d epochs):" % EPOCHS)
    for i, (name, best, final, dt) in enumerate(sorted(results, key=lambda r: -r[1]), 1):
        print("  %d. %-34s %.4f%%" % (i, name, best * 100))


if __name__ == "__main__":
    main()
