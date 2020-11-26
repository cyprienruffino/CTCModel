import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

        
setuptools.setup(
    name="keras_ctcmodel",
    version="1.0.0rc3",
    install_requires=install_requires,
    author="Cyprien Ruffino",
    author_email="ruffino.cyprien@gmail.com",
    description="Easy-to-use Connectionnist Temporal Classification in Keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cyprienruffino/CTCModel",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires='>=3.5',
)
