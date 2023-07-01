
# Genetic Algorithm for Neural Network Weight Optimization
This repository contains a program that uses a genetic algorithm to improve the weights of a neural network model. The goal of the program is to train the neural network to correctly classify binary strings into either 0 or 1.

Repository Structure
The repository is organized as follows:

* buildent0_1 + buildnet0_2: These two zip directories contain the files required to run the buildnet0.exe program. To execute the buildnet0.exe, you need to extract both zip files and combine them into a single directory. The purpose of buildnet0.exe is to find the optimal weights for predicting the dataset "nn0".

* buildent1_1 + buildnet1_2: Similar to the previous set of directories, these contain the files necessary to run the buildnet1.exe program. Once again, you should extract both zip files and merge them into one directory. The buildnet1.exe program aims to find the correct weights for predicting the dataset "nn1".

* runnet0.zip: This zip directory contains the feedforward neural network weights that were discovered during the execution of buildnet0.exe over a new dataset called "testnet0". The program utilizes the weights found in the wnet0.json file, which is the output of buildnet0.exe.

* runnet1.zip: Similar to the previous directory, this zip contains the feedforward neural network weights obtained from running buildnet1.exe over the "testnet1" dataset. The necessary network configuration is stored in the wnet1.json file.

