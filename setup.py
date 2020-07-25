import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyCESat2"
    version="0.0.1",
    author="Forrest Corcoran",
    author_email="fcorcora@oregonstate.edu",
    description="Python Library for ICESat-2 processing and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fpcorcoran/PyCESat2",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        #"License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6.7',
)
