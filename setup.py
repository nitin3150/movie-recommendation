# setup.py
from setuptools import setup, find_packages

setup(
    name="recommender",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'tqdm'
    ]
)