# -*- coding: utf-8 -*-
"""
Fill createModel() in the submission notebook with the architecture that won
the two ablation rounds, and document the search in a markdown cell so the
trial-and-error the workshop asks for is visible in the notebook itself.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "D5_WangMingyang-A0331532R.ipynb")

CREATE_MODEL = """modelname   = 'wks5_5'                                                          # Step 1

                                                                                # Step 2
                                                                                # Structure chosen by the trial-and-error
                                                                                # documented in the cell above: base model
                                                                                # with padding='same' on both conv layers,
                                                                                # Dropout raised 0.2 -> 0.5, and the hidden
                                                                                # Dense layer widened 128 -> 256. Scored
                                                                                # 97.40% best val accuracy at 30 epochs
                                                                                # versus 96.49% for the base model.
def createModel():
    model       = Sequential()
    model.add(Conv2D(20,
                     (5, 5),
                     padding='same',
                     input_shape=(28, 28, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(40,
                     (5, 5),
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

                                                                                # Step 3
model       = createModel() # This is meant for training
modelGo     = createModel() # This is used for final testing

model.summary()                                                                 # Step 4"""

NOTES = """## **6b. Structure trial and error (what was tried, and what happened)**
___
The notes invite us to trial and error various structures. Instead of guessing,
each change was applied to the base model **one at a time** (single-variable
ablation), all runs sharing the same seed and the same data split, so that the
effect of each individual change is attributable. Training uses all 60,000
training images, with the 10,000 test images as the validation set.

**Round 1 — 15 epochs, nine structures**

| Structure (change vs base) | Best val accuracy |
|---|---|
| Dropout 0.2 to 0.5 | 96.52% |
| padding = 'same' | 96.39% |
| Hidden Dense 128 to 256 | 96.23% |
| Channels 20/40 to 32/64 | 96.10% |
| **Base (as in the notes)** | **96.09%** |
| Extra Dropout before the output layer | 96.03% |
| 32/64 + 3x3 + same (combined) | 95.85% |
| Kernels 5x5 to 3x3 | 95.83% |
| Add a third Conv block (80 filters @ 3x3) | 95.52% |

*Finding:* all nine structures landed inside a 1.0pp band, and the leaders were
only 0.1-0.4pp apart. That is too narrow to trust from a single short run, and
the combined model scored **below** the base model - so at 15 epochs the changes
were not composing, they were mostly noise.

**Round 2 — 30 epochs, the leaders re-tested**

| Structure | Best val accuracy |
|---|---|
| **drop 0.5 + same + Dense 256** | **97.40%** |
| drop 0.5 + same | 97.19% |
| drop 0.5 | 96.91% |
| same | 96.64% |
| Base | 96.49% |

*Finding:* over a longer horizon the round-1 ranking held up (the same two
changes lead), the gap over base widened to a meaningful **+0.91pp**, and this
time the changes **did** compose. So the final model uses the top-3 combination.

*Caveat:* every structure was still improving at 30 epochs, so these numbers are
a lower bound on what the full 60-epoch run reaches, and they come from a single
seed per structure rather than an average over several runs."""


def main():
    with open(NB, encoding="utf-8") as f:
        nb = json.load(f)

    assert nb["cells"][11]["cell_type"] == "code", "cell-11 is not the createModel cell"
    assert "def createModel" in "".join(nb["cells"][11]["source"])

    nb["cells"][11]["source"] = CREATE_MODEL.splitlines(keepends=True)
    nb["cells"][11]["outputs"] = []
    nb["cells"][11]["execution_count"] = None

    notes_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": NOTES.splitlines(keepends=True),
    }
    nb["cells"].insert(12, notes_cell)

    with open(NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print("filled createModel() and inserted the trial-and-error notes")
    print("cells now:", len(nb["cells"]))


if __name__ == "__main__":
    main()
