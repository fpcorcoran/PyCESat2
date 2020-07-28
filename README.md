# PyCESAT2


## Summary

PyCESAT2 is a python library designed to read, process, and visualize ATL03 data
from the ATLAS space-borne LiDAR sensor onboard NASA's ICESat-2 satellite.

**This package is still highly developmental and has not been approved for release.**

## Installation

Due to the developmental nature of this package, it is not yet available via
`conda install` or `pip install`

Instead, please fork and clone this repository. From within the cloned directory...

### **conda users:**

1. Create a new virtual environment with `conda env create -f environment.yaml`

2. Add PyCESat2 to virtual environment with `conda develop .`

### **pip users:**

1. Create a new virtual environment with *venv* - https://docs.python.org/3/tutorial/venv.html

2. Install package requirements in new environment with `pip install -r requirements.txt`

3. Add PyCESat2 to virtual environment with `pip install .`

## Data Access

The preferred method for accessing ICESat-2 ATLAS ATL03 data is via
https://openaltimetry.org/data/icesat2/.

Alternatively, data can be accessed via the National Snow and Ice Data Center
(NSIDC) website:
https://nsidc.org/data/ATL03.

*Important Note:* PyCESAT2 is designed specifically for ATL03 data and does not
support the hierarchical structure of other ATLAS data products. Future releases
may be designed for these products, but at this time they are not a priority.

## Contributing

Given that this repo is currently private, if you're reading this you already
have permission to contribute.

*Rules of Thumb/Best Practices*

* Please fork the repository in order to copy it to your own Github account.

* For any contributions, please follow these protocol:
	1. Create a new issue (with label) detailing your planned addition
	2. Create a new branch of form #-Name-Of-Issue.
		- Number should be the next number after the highest numbered branch (active or inactive)
	3. Issue pull request with changes to new branch and assign Forrest Corcoran as reviewer
