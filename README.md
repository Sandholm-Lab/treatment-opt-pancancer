## Treatment Optimization Pan-cancer

This project uses optimization algorithm to discover efficicent multi-drug combinations to treat various lines of cancer cells.
Fabian Fröhlich's (https://github.com/FFroehlich/reference_simulator) pan-cancer pathway model is employed to predict the efficiency of drug combinations

To run this software the following steps are required:

* Clone this repository.
* Clone the code for the simulator (https://github.com/FFroehlich/reference_simulator) into src/reference_simulator/.
* Run src/reference_simulator/build_model.py to create the necessary binarys to run the simulator. This might require the installation of multiple Python packages.
* Check if the software is running as expected by executing: python -m unittest discover /test -v

A list with required dependencies is provided in /other. 