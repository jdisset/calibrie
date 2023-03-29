
from setuptools import setup, find_packages

setup(
    name="calibry",
    version="0.1.0",
    author="Jean Disset",
    author_email="jdisset@mit.edu",
    description="Calibry: flow cytometry calibration package",
    packages=find_packages(),
    install_requires=[
        "jaxlib",
        "jax[cpu]",
        "ott-jax",
        "optax",
        "scipy",
        "pandas",
        "numpy",
        "matplotlib",
        "flowio",
        "tqdm"
    ],
)
