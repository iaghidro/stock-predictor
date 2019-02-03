#!/bin/bash

echo "INSTALL PYTHON DEPENDENCIES";
pip3 install ipython==5.3.0
pip3 install pytest
pip3 install ta
pip3 install sklearn
pip3 install pandas
pip3 install jupyter
pip3 install -U feather-format
pip3 install matplotlib
pip3 install 'prompt-toolkit==1.0.15'
export PATH=$PATH:~/.local/bin/

echo "INSTALL DEPENDENCIES";
npm i;

echo "RUN UNIT TESTS";
npm test;
