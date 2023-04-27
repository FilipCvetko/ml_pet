import datetime
import os
import pickle
import re
import warnings

import numpy as np
import pydicom
from pydicom.data import get_testdata_file
from pydicom.errors import InvalidDicomError
from pydicom.filereader import read_dicomdir
from tqdm import tqdm
from utils import interpolate_ct_dict


def create_3d_array(input_dict):
    # Find the shape of the 3D numpy array
    z_len = len(input_dict)
    x_len = max(len(v) for v in input_dict.values())
    y_len = max(len(v[0]) for v in input_dict.values())

    # Create an empty 3D numpy array with the determined shape
    arr = np.zeros((z_len, x_len, y_len))

    # Fill in the values from the input dictionary
    for i, (_, value) in enumerate(sorted(input_dict.items())):
        arr[i, : len(value), : len(value[0])] = value

    return arr


def get_full_ct_dict(
    patient_folder, required_series_keywords, required_study_keywords=[]
):
    ct_dict = {}
    for file in os.listdir(patient_folder):
        if not file.endswith(".IMA"):
            continue
        try:
            ds = pydicom.dcmread(f"{patient_folder}/{file}")
        except InvalidDicomError:
            return ct_dict
        if required_study_keywords != []:
            if not all([x in ds.StudyDescription for x in required_study_keywords]):
                continue
        if not all([x in ds.SeriesDescription for x in required_series_keywords]):
            continue

        ct_dict[float(ds.ImagePositionPatient[2])] = ds.pixel_array

    return ct_dict


def generate_true(
    save_folder="/home/filip/IT/Projects/ml_pet/data/interim/",
    required_study_keywords=["2faza"],
    required_series_keywords=["2mm", "OSEM"],
    generate_ct=False,
    required_ct_series_keywords=["CT", "AC"],
):
    for patient_folder in tqdm(os.listdir(os.getenv("DICOM_DATA_DIR"))):
        print(f"Checking {patient_folder}")
        if patient_folder.startswith("ctk"):
            continue

        if not re.match(r"\d+[A-Z]{2,5}", patient_folder):
            raise ValueError(
                f"Patient folder {patient_folder} may be invalid. Check if the folder is valid and change the name so that the folder name starts with a digit followed by first and last name acronym."
            )
        current_dir = os.getenv("DICOM_DATA_DIR") + "/" + patient_folder

        pet_dict = {}
        if generate_ct:
            full_ct_dict = get_full_ct_dict(
                patient_folder=current_dir,
                required_series_keywords=required_ct_series_keywords,
                required_study_keywords=required_study_keywords,
            )
            if full_ct_dict == {}:
                print(
                    f"Skipping control patient {patient_folder} due to invalid CT slices."
                )
                continue
            ct_dict = {}

        for file in os.listdir(current_dir):
            if not file.endswith(".IMA"):
                continue
            ds = pydicom.dcmread(f"{current_dir}/{file}")
            if not all([x in ds.StudyDescription for x in required_study_keywords]):
                continue
            if not all([x in ds.SeriesDescription for x in required_series_keywords]):
                continue

            pet_dict[str(ds.ImageIndex)] = ds.pixel_array
            if generate_ct:
                ct_dict[str(ds.ImageIndex)] = interpolate_ct_dict(
                    full_ct_dict, float(ds.ImagePositionPatient[2])
                )

        # If the configuration is not found, reject the sample.
        if not pet_dict:
            continue

        out_pet_array = create_3d_array(pet_dict)
        if generate_ct:
            out_ct_array = create_3d_array(ct_dict)

        # Now create the appropriate folders if they're not already inplace.
        now = datetime.datetime.now()
        date_time_str = now.strftime("%d-%m-%Y")

        if not os.path.exists(save_folder + "/" + date_time_str):
            os.mkdir(save_folder + "/" + date_time_str)

        unique_identifier = (
            "_".join(required_study_keywords) + "_" + "_".join(required_series_keywords)
        )
        subfolder = save_folder + "/" + date_time_str + "/" + unique_identifier
        if not os.path.exists(subfolder):
            os.mkdir(subfolder)

        # Now save the contents to the folder with the appropriate setting
        # identifying patients based on PatientName DICOM property
        with open(subfolder + "/" + str(ds.PatientName), "wb") as f:
            pickle.dump(out_pet_array, f)

        if generate_ct:
            with open(subfolder + "/" + str(ds.PatientName) + "_CT", "wb") as f:
                pickle.dump(out_ct_array, f)


def generate_controls(
    save_folder="/home/filip/IT/Projects/ml_pet/data/interim/",
    required_series_keywords=["2mm", "OSEM"],
    required_ct_series_keywords=["CT", "rostat", "TERATIVN"],
    generate_ct=False,
):
    for patient_folder in tqdm(os.listdir(os.getenv("DICOM_CONTROLS_DATA_DIR"))):
        print(f"Checking {patient_folder}")
        if patient_folder.startswith("ctk"):
            continue

        if not re.match(r"\d+[A-Z]{2,5}", patient_folder):
            raise ValueError(
                f"Patient folder {patient_folder} may be invalid. Check if the folder is valid and change the name so that the folder name starts with a digit followed by first and last name acronym."
            )
        current_dir = os.getenv("DICOM_CONTROLS_DATA_DIR") + "/" + patient_folder

        pet_dict = {}
        if generate_ct:
            full_ct_dict = get_full_ct_dict(
                patient_folder=current_dir,
                required_series_keywords=required_ct_series_keywords,
            )
            if full_ct_dict == {}:
                print(
                    f"Skipping control patient {patient_folder} due to invalid CT slices."
                )
                continue
            ct_dict = {}

        for file in os.listdir(current_dir):
            if not file.endswith(".IMA"):
                continue

            try:
                ds = pydicom.dcmread(f"{current_dir}/{file}")
            except:
                continue

            if not all([x in ds.SeriesDescription for x in required_series_keywords]):
                continue

            pet_dict[str(ds.ImageIndex)] = ds.pixel_array
            if generate_ct:
                ct_dict[str(ds.ImageIndex)] = interpolate_ct_dict(
                    full_ct_dict, float(ds.ImagePositionPatient[2])
                )

        # If the configuration is not found, reject the sample.
        if not pet_dict:
            continue

        out_pet_array = create_3d_array(pet_dict)
        if generate_ct:
            out_ct_array = create_3d_array(ct_dict)

        # Now create the appropriate folders if they're not already inplace.
        now = datetime.datetime.now()
        date_time_str = now.strftime("%d-%m-%Y")

        if not os.path.exists(save_folder + "/" + date_time_str):
            os.mkdir(save_folder + "/" + date_time_str)

        unique_identifier = "controls_" + "_".join(required_series_keywords)
        subfolder = save_folder + "/" + date_time_str + "/" + unique_identifier
        if not os.path.exists(subfolder):
            os.mkdir(subfolder)

        # Now save the contents to the folder with the appropriate setting
        # identifying patients based on PatientName DICOM property
        with open(subfolder + "/" + str(ds.PatientName), "wb") as f:
            pickle.dump(out_pet_array, f)

        if generate_ct:
            with open(subfolder + "/" + str(ds.PatientName) + "_CT", "wb") as f:
                pickle.dump(out_ct_array, f)


generate_controls(
    save_folder="/home/filip/IT/Projects/ml_pet/data/interim/",
    required_series_keywords=["4mm", "OSEM"],
    generate_ct=True,
)

# generate_true(
#     save_folder="/home/filip/IT/Projects/ml_pet/data/interim/",
#     required_study_keywords=["2faza"],
#     required_series_keywords=["4mm", "OSEM"],
#     generate_ct=True
# )
