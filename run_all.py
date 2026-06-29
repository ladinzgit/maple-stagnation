"""
Run all analysis notebooks in order.
Requires: pip install -r requirements.txt
Usage:    python run_all.py
"""
import subprocess
import sys

NOTEBOOKS = [
    "h1_clustering/feature_selection.ipynb",
    "h1_clustering/h1_clustering.ipynb",
    "h2_distribution/h2_distribution.ipynb",
    "h3_rule/h3_rule.ipynb",
]

def run_notebook(path: str) -> None:
    print(f"\n>>> {path}")
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook", "--execute", "--inplace", path,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[FAILED] {path}")
        sys.exit(result.returncode)
    print(f"[OK] {path}")

def main() -> None:
    subprocess.run(
        ["python", "h2_distribution/run_analysis.py"], check=True
    )
    for nb in NOTEBOOKS:
        run_notebook(nb)
    print("\nAll analyses complete.")

if __name__ == "__main__":
    main()
