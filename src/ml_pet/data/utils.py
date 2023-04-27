import os
import re

import numpy as np
import pydicom


def get_largest_wb_pet_thyroid_slice_number(
    required_series_keywords=["2mm", "OSEM"],
):
    largestNum = 0

    for patient_folder in os.listdir(os.getenv("DICOM_CONTROLS_DATA_DIR")):
        print(f"Checking {patient_folder}")
        if patient_folder.startswith("ctk"):
            continue

        patient_dict = {}

        if not re.match(r"\d+[A-Z]{2,5}", patient_folder):
            raise ValueError(
                f"Patient folder {patient_folder} may be invalid. Check if the folder is valid and change the name so that the folder name starts with a digit followed by first and last name acronym."
            )
        current_dir = os.getenv("DICOM_CONTROLS_DATA_DIR") + "/" + patient_folder
        for file in os.listdir(current_dir):
            if not file.endswith(".IMA"):
                continue

            try:
                ds = pydicom.dcmread(f"{current_dir}/{file}")
            except:
                continue

            if not all([x in ds.SeriesDescription for x in required_series_keywords]):
                continue

            if ds.NumberOfSlices > largestNum:
                largestNum = ds.NumberOfSlices

    return largestNum


def interpolate_ct_dict(my_dict, key):
    keys = sorted(my_dict.keys())
    if key in keys:
        return my_dict[key]
    if key < keys[0]:
        return my_dict[keys[0]]
    if key > keys[-1]:
        return my_dict[keys[-1]]
    left_key, right_key = (
        keys[np.searchsorted(keys, key) - 1],
        keys[np.searchsorted(keys, key)],
    )
    t = (key - left_key) / (right_key - left_key)
    return my_dict[left_key] * (1 - t) + my_dict[right_key] * t
