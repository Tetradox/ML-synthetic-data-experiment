# ML-synthetic-data-experiment
This repository contains the code and the data used for testing the machine learning models quality using synthetic data augmentation
## Repository contents
1. synthetic.py - main model, which contains 4 different synthesizers, 4 machine learning models as well as hyperparameter tuning using RandomizedSearchCV and 3 different model quality metrics.
2. clean_data.csv - data used for modeling
3. requirements.txt - all of the libraries used in the main script
4. LICENSE - Apache-2.0
## Input data
Dataset used in this project contains 4 descriptors: I_D/I_G, Temperature, Time, BET and 1 output parameter: Selectivity. There's a total of 40 unique rows of data
## Installation
1. Create a virtual environment
2. Type in your terminal: "pip install -r requirements.txt"
## Python environment
The version of python used for this pipeline is 3.14.3
## Script usage example
1. Prepare your dataset: it should be in the form of a csv file, needs to have no empty rows and considering that this script uses regression models, the output parameter should be non binary
2. Rename the input and output columns in the synthetic.py file so they match your data
3. Start the script using "synthetic.py your_data.csv"
4. Wait for a bit and get your results!
## Expected outputs
The script should output a dictionary of results containing top-5 best models by R2
## License
The source code in this repository is released under the Apache License 2.0. See LICENSE.
