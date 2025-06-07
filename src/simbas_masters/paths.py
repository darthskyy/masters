# Just some path management for the different paths in this project.
from pathlib import Path
from typing import Union
import os

BASE_DIR = Path(__file__).resolve().parent  # Assuming this file is in src/simbas_masters
PROJECT_ROOT = BASE_DIR.parent  # Assuming the project root is one level up from src
DATA_DIR = BASE_DIR / "_data"
UTILS_DIR = BASE_DIR / "_utils"
OUTPUT_DIR = DATA_DIR / "out"
def ensure_data_dir_exists():
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ensure_data_dir_exists()