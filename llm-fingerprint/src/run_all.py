"""Run every CNN/LSTM experiment: 2 models x 3 modes x 5 seeds.
Usage:  python src/run_all.py [dataset | dataset_mismatch]
"""
import sys
import itertools
import subprocess
from config import SEEDS, ROOT

dataset = sys.argv[1] if len(sys.argv) > 1 else "dataset"
for seed, model, mode in itertools.product(SEEDS, ["cnn", "lstm"], ["output", "input", "both"]):
    subprocess.run([sys.executable, str(ROOT / "src" / "train.py"), "--model", model,
                    "--mode", mode, "--seed", str(seed), "--dataset", dataset], check=True)
