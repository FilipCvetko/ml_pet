from setuptools import find_packages, setup
from typing import List

import io
import os

def parse_requirements(filename: str) -> List[str]:
    required_packages = []
    with open(os.path.join(os.path.dirname(__file__), filename)) as req_file:
        for line in req_file:
            required_packages.append(line.strip())
    return required_packages

setup(
    name='ml_pet',
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=parse_requirements("requirements.txt"),
    extras_requires=parse_requirements("requirements-dev.txt"),
    version='0.1.0',
    description='A python repository of research and utility code for analysis of PET image reconstructions and automatic detection of parathyroid adenomas from [F-18]-FCH PET/CT',
    author='Filip Cvetko',
    license='MIT',
)
