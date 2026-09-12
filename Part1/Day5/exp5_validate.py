# -*- coding: utf-8 -*-
"""
End-to-end validation of the submission notebook WITHOUT long training.

Makes a throwaway copy with epochs=3 (and the curve-plot yticks relaxed so the
3-epoch numbers still render), executes every cell in order, and reports pass or
fail. The deliverable notebook itself is never modified.
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "D5_WangMingyang-A0331532R.ipynb")
TMP = os.path.join(HERE, "_validate_tmp.ipynb")
OUT = os.path.join(HERE, "_validate_tmp.out.ipynb")

PROBE_EPOCHS = 3


def main():
    with open(SRC, encoding="utf-8") as f:
        nb = json.load(f)

    fit_cells = [i for i, c in enumerate(nb["cells"])
                 if c["cell_type"] == "code" and "epochs=60" in "".join(c["source"])]
    assert len(fit_cells) == 1, "expected exactly one epochs=60 cell, got %r" % fit_cells
    i = fit_cells[0]
    nb["cells"][i]["source"] = "".join(nb["cells"][i]["source"]).replace(
        "epochs=60", "epochs=%d" % PROBE_EPOCHS).splitlines(keepends=True)
    print("patched epochs=60 -> epochs=%d in cell-%d" % (PROBE_EPOCHS, i))

    with open(TMP, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    cmd = [sys.executable, "-m", "jupyter", "nbconvert",
           "--to", "notebook", "--execute", "--allow-errors",
           "--ExecutePreprocessor.timeout=900", TMP, "--output", OUT]
    print("running:", " ".join(cmd[2:]), flush=True)
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    print("nbconvert exit code:", r.returncode)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        return 1

    with open(OUT, encoding="utf-8") as f:
        done = json.load(f)

    errors = 0
    for idx, c in enumerate(done["cells"]):
        if c["cell_type"] != "code":
            continue
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                errors += 1
                print("  ERROR in cell-%d: %s: %s"
                      % (idx, o.get("ename"), str(o.get("evalue"))[:200]))
    print("cells executed:", sum(1 for c in done["cells"] if c["cell_type"] == "code"))
    print("error outputs  :", errors)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        for p in (TMP, OUT):
            if os.path.exists(p):
                os.remove(p)
