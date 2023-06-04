import os

import numpy as np
import pydicom
from pydicom.data import get_testdata_file
from pydicom.filereader import read_dicomdir

directory = os.getenv("TEST_DICOM_IMA_SLICE")

ds = pydicom.dcmread(directory)


print("Patient Name:", ds.PatientName)
print("Patient ID:", ds.PatientID)

# Print modality and series number
print("Modality:", ds.Modality)
print("Series Number:", ds.SeriesNumber)

# Print image position and pixel spacing
print("Image Position:", ds.ImagePositionPatient)
print("Pixel Spacing:", ds.PixelSpacing)

# Print image dimensions
print("Image Size:", ds.Rows, "x", ds.Columns)

# Print Study Description
print("Study: ", ds.StudyDescription)

# Print raw pixel data
print(ds.pixel_array)
