import json, sys
sys.stdout.reconfigure(encoding="utf-8")

for nb_path, indices in [("eda/eda.ipynb", [2, 3, 9, 43]), ("h1_clustering/h1_clustering.ipynb", [0, 1])]:
    print(f"\n=== {nb_path} ===")
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    for i in indices:
        cell = nb["cells"][i]
        cid = cell.get("id") or cell.get("metadata", {}).get("id", "NONE")
        print(f"Cell {i}: id={cid!r}")
