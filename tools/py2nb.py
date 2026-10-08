"""'# %%' / '# %% [markdown]' script -> .ipynb"""
import json, re, sys, hashlib
src = open(sys.argv[1]).read()
cells = []
for block in re.split(r"^# %%", src, flags=re.M)[1:]:
    head, _, body = block.partition("\n")
    body = body.strip("\n")
    if "[markdown]" in head:
        md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        cells.append({"cell_type": "markdown", "id": hashlib.md5(f"{len(cells)}".encode()).hexdigest()[:8], "metadata": {}, "source": md.splitlines(True)})
    else:
        cells.append({"cell_type": "code", "id": hashlib.md5(f"{len(cells)}".encode()).hexdigest()[:8], "metadata": {}, "execution_count": None, "outputs": [], "source": body.splitlines(True)})
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
      "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
print(len(cells), "cells")
