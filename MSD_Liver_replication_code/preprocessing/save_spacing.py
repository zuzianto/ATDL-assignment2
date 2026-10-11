import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
import pickle
from tqdm import tqdm
from scipy.ndimage import zoom


# 定义原始数据和目标路径
def get_promise12_spacing():

    RAW_DATA_DIR = "../data/Promise12/raw_data"

    Foreslices_IMAGE_DIR = "../data/Promise12/Foreslices/image"
    Foreslices_LABEL_DIR = "../data/Promise12/Foreslices/label"
    Foreslices_spacing_DIR = "../data/Promise12/Foreslices/spacing"


    for dir_path in [
        Foreslices_IMAGE_DIR,
        Foreslices_LABEL_DIR,
        Foreslices_spacing_DIR,
    ]:
        os.makedirs(dir_path, exist_ok=True)

    global_idx = 0
    alls = ("training_data", "livechallenge_test_data", "test_data")
    a = []
    for s in alls:
        base_dir = os.path.join(RAW_DATA_DIR, s)
        files = [
            os.path.join(base_dir, f)
            for f in os.listdir(base_dir)
            if f.endswith("mhd") and "seg" not in f
        ]
        files.sort()
        for file in tqdm(files):
            image = sitk.ReadImage(file)
            spacing_path = os.path.join(
                Foreslices_spacing_DIR,
                f"Case{global_idx:03d}_spacing.pkl",
            )
            with open(spacing_path, "wb") as f:
                pickle.dump(image.GetSpacing(), f)


            

    
            global_idx += 1

def get_liver_spacing():
    """Store spacing in (z,y,x) order for volumes resized in-plane to 512x512."""
    raw_dir = "../data/Liver/raw/Task03_Liver/imagesTr"
    spacing_dir = "../data/Liver/Foreslices/spacing"
    os.makedirs(spacing_dir, exist_ok=True)
    for file in tqdm(sorted(os.listdir(raw_dir)), ncols=100):
        if not file.endswith(".nii.gz") or not file.startswith("liver_"):
            continue
        idx = int(file.split("_")[1].split(".")[0])
        image = sitk.ReadImage(os.path.join(raw_dir, file))
        sx, sy, sz = image.GetSpacing()  # SimpleITK order: x, y, z
        nx, ny, nz = image.GetSize()
        metadata = {
            "original_spacing_xyz": (sx, sy, sz),
            "original_size_xyz": (nx, ny, nz),
            "resized_spacing_zyx": (sz, sy * ny / 512.0, sx * nx / 512.0),
        }
        path = os.path.join(spacing_dir, f"Case{idx:03d}_spacing.pkl")
        with open(path, "wb") as f:
            pickle.dump(metadata, f)


if __name__ == "__main__":
    # get_promise12_spacing()
    get_liver_spacing()


        



