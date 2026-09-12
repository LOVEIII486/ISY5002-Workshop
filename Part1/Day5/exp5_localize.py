# -*- coding: utf-8 -*-
"""
Build the local-execution submission notebook from the Colab template.

Every replacement is asserted, so a silent no-op cannot slip through
if the template text ever changes.
"""
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "psupr_wks5_5_yourname_.ipynb")
DST = os.path.join(HERE, "D5_WangMingyang-A0331532R.ipynb")

DRIVE_DATA = "/content/gdrive/My Drive/iss/psupr/data/"
DRIVE_OUT = "/content/gdrive/My Drive/iss/psupr/colab/"


def replace_in_cell(nb, idx, old, new, expect=1):
    src = "".join(nb["cells"][idx]["source"])
    n = src.count(old)
    assert n == expect, "cell-%d: expected %d occurrence(s) of %r, found %d" % (
        idx, expect, old, n)
    nb["cells"][idx]["source"] = src.replace(old, new).splitlines(keepends=True)
    return n


def main():
    shutil.copyfile(SRC, DST)
    with open(DST, encoding="utf-8") as f:
        nb = json.load(f)

    # 1) matplotlib style: 'seaborn' was removed in matplotlib 3.6+
    #    the lecturer's own .py uses 'ggplot'
    replace_in_cell(nb, 7, "plt.style.use('seaborn')", "plt.style.use('ggplot')")

    # 2) data files now live next to the notebook (4 np.load calls in cell-9)
    replace_in_cell(nb, 9, DRIVE_DATA, "", expect=4)

    # 3) output folder: write next to the notebook instead of into Drive
    replace_in_cell(nb, 13, "folderpath      = '%s'" % DRIVE_OUT,
                    "folderpath      = ''")

    # 4) the template ships Colab metadata; 'python3' resolves to a local
    #    interpreter WITHOUT tensorflow, so pin the notebook to the conda env
    nb["metadata"]["kernelspec"] = {
        "name": "isy5002-py310",
        "display_name": "Python (isy5002-py310)",
    }
    nb["metadata"].pop("accelerator", None)  # was "GPU" - untrue locally

    with open(DST, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    # verify nothing Colab-specific is left
    left = [d for d in (DRIVE_DATA, DRIVE_OUT, "style.use('seaborn')")
            if d in json.dumps(nb)]
    print("wrote:", os.path.basename(DST))
    print("remaining Colab/Drive references:", left if left else "none")


if __name__ == "__main__":
    main()
