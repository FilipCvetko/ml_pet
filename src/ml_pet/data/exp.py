import pickle

import matplotlib.pyplot as plt
import numpy as np

with open(
    "/home/filip/IT/Projects/ml_pet/data/interim/02-04-2023/1faza_4mm_OSEM/1KM", "rb"
) as file:
    arr = pickle.load(file)

plt.imshow(arr[66, :, :], cmap="gray")
plt.show()
