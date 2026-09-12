# -*- coding: utf-8 -*-
"""
Day5 workshop - architecture search.

Single-variable ablation: every config differs from the lecturer's base model
in exactly ONE place, so we can tell which change actually helps.

Each config trains for EPOCHS epochs (short) with the same seed, and we report
the best validation accuracy reached. This is a parameter probe, not the final
training run.
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
EPOCHS = 15
BATCH = 128


def load_data():
    trDat = np.load(os.path.join(HERE, "kmnist-train-imgs.npz"))["arr_0"]
    trLbl = np.load(os.path.join(HERE, "kmnist-train-labels.npz"))["arr_0"]
    tsDat = np.load(os.path.join(HERE, "kmnist-test-imgs.npz"))["arr_0"]
    tsLbl = np.load(os.path.join(HERE, "kmnist-test-labels.npz"))["arr_0"]

    trDat = (trDat.astype("float32") / 255).reshape(-1, 28, 28, 1)
    tsDat = (tsDat.astype("float32") / 255).reshape(-1, 28, 28, 1)

    return trDat, to_categorical(trLbl), tsDat, to_categorical(tsLbl)


# ---------------------------------------------------------------- configs
# base = lecturer's structure; every other entry changes ONE thing.

def cfg_base(nc):
    m = Sequential()
    m.add(Conv2D(20, (5, 5), input_shape=(28, 28, 1), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(40, (5, 5), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.2))
    m.add(Flatten())
    m.add(Dense(128, activation="relu"))
    m.add(Dense(nc, activation="softmax"))
    return m


def cfg_wide(nc):
    """more channels: 20/40 -> 32/64"""
    m = Sequential()
    m.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(64, (5, 5), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.2))
    m.add(Flatten())
    m.add(Dense(128, activation="relu"))
    m.add(Dense(nc, activation="softmax"))
    return m


def cfg_k3(nc):
    """smaller kernels: 5x5 -> 3x3"""
    m = Sequential()
    m.add(Conv2D(20, (3, 3), input_shape=(28, 28, 1), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(40, (3, 3), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.2))
    m.add(Flatten())
    m.add(Dense(128, activation="relu"))
    m.add(Dense(nc, activation="softmax"))
    return m


def cfg_same(nc):
    """padding='same' instead of no padding"""
    m = Sequential()
    m.add(Conv2D(20, (5, 5), padding="same", input_shape=(28, 28, 1), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(40, (5, 5), padding="same", activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.2))
    m.add(Flatten())
    m.add(Dense(128, activation="relu"))
    m.add(Dense(nc, activation="softmax"))
    return m


def cfg_drop05(nc):
    """dropout rate 0.2 -> 0.5"""
    m = Sequential()
    m.add(Conv2D(20, (5, 5), input_shape=(28, 28, 1), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(40, (5, 5), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.5))
    m.add(Flatten())
    m.add(Dense(128, activation="relu"))
    m.add(Dense(nc, activation="softmax"))
    return m


def cfg_drop2nd(nc):
    """add a 2nd dropout between the two Dense layers"""
    m = Sequential()
    m.add(Conv2D(20, (5, 5), input_shape=(28, 28, 1), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(40, (5, 5), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.2))
    m.add(Flatten())
    m.add(Dense(128, activation="relu"))
    m.add(Dropout(0.5))
    m.add(Dense(nc, activation="softmax"))
    return m


def cfg_dense256(nc):
    """hidden dense layer 128 -> 256"""
    m = Sequential()
    m.add(Conv2D(20, (5, 5), input_shape=(28, 28, 1), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(40, (5, 5), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.2))
    m.add(Flatten())
    m.add(Dense(256, activation="relu"))
    m.add(Dense(nc, activation="softmax"))
    return m


def cfg_conv3(nc):
    """3 conv blocks: 20/40/80 instead of 20/40"""
    m = Sequential()
    m.add(Conv2D(20, (5, 5), input_shape=(28, 28, 1), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(40, (5, 5), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(80, (3, 3), activation="relu"))
    m.add(Dropout(0.2))
    m.add(Flatten())
    m.add(Dense(128, activation="relu"))
    m.add(Dense(nc, activation="softmax"))
    return m


def cfg_wide_k3(nc):
    """combo: 32/64 channels + 3x3 kernels + same padding"""
    m = Sequential()
    m.add(Conv2D(32, (3, 3), padding="same", input_shape=(28, 28, 1), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.2))
    m.add(Flatten())
    m.add(Dense(128, activation="relu"))
    m.add(Dense(nc, activation="softmax"))
    return m


CONFIGS = [
    ("base       (lecturer 20/40, 5x5, no-pad)", cfg_base),
    ("wide       32/64 channels", cfg_wide),
    ("k3         3x3 kernels", cfg_k3),
    ("same       padding='same'", cfg_same),
    ("drop0.5    dropout 0.2->0.5", cfg_drop05),
    ("drop2nd    +dropout before output", cfg_drop2nd),
    ("dense256   hidden 128->256", cfg_dense256),
    ("conv3      +3rd conv block 80@3x3", cfg_conv3),
    ("wide+k3+same  combined", cfg_wide_k3),
]


def main():
    trDat, trLbl, tsDat, tsLbl = load_data()
    nc = trLbl.shape[1]
    print("train", trDat.shape, "test", tsDat.shape, "classes", nc, flush=True)
    print("%-38s %10s %9s %8s" % ("config", "best_val", "final_val", "secs"))
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
        best, final = max(vals), vals[-1]
        results.append((name, best, final, dt))
        print("%-38s %9.4f%% %8.4f%% %7.0fs"
              % (name, best * 100, final * 100, dt), flush=True)

    print("-" * 70, flush=True)
    ranked = sorted(results, key=lambda r: -r[1])
    print("RANKING by best val_accuracy (%d epochs each):" % EPOCHS)
    for i, (name, best, final, dt) in enumerate(ranked, 1):
        print("  %d. %-36s %.4f%%" % (i, name, best * 100))


if __name__ == "__main__":
    main()
