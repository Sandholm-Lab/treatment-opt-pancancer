## Treatment Optimization Pan-cancer

This project uses optimization algorithm to discover efficicent multi-drug combinations to treat various lines of cancer cells.
Fabian Fröhlich's (https://github.com/FFroehlich/reference_simulator) pan-cancer pathway model is employed to predict the efficiency of drug combinations

To run this software the following steps are required:

* Clone this repository.
* Copy the code for the simulator (https://github.com/FFroehlich/reference_simulator) into src/reference_simulator/.
* Install the following Python packages: 
```
pip install python-libsbml
pip install amici
pip install petab==0.0.0a16
pip install gym
```
* Run src/reference_simulator/build_model.py to create the necessary binarys to run the simulator (this takes arround 1h).
* Check if the software is running as expected by executing: python -m unittest discover /test -v

A list with required dependencies is provided in /other. 

The source code contained in the folder src/sampler is based on Ari Pakman's GitHub repository https://github.com/aripakman/hmc-tmg and is subject to a MIT License.
