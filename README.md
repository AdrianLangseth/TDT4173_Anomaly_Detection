# TDT4173_Anomaly_Detection

## Bayesian Neural Network (BNN)

A simple Bayesian neural network that classifies MNIST images. The network consists of one input layer with 784 (28*28) nodes, one hidden layer with 512 nodes, and an output layer with 10 nodes. 

With a training set size of 50 000 and 1000 epochs of training, it has around 92% accuracy. 

### Project Structure
Code relating to the BNN can be found in `src/bnn/`. The folder contains the following files:
* `Net.py` Contains the code for the BNN itself. 
* `settings.py` Defines various constants and variables relating to the project.
* `data.py` Contains code for loading and processing the MNIST and NotMNIST datasets. 
* `main.py` Functions for setting up the environment for training networks. Most of the hyperparameters are defined here. 
* `interface.py` Interface to the BNN that is used to easily access metrics about the network's performance.
* `website_data.py` Contains a small snippet of code for retriving the BNN's prediction entropy for the 10 instances with highest FFNN prediction entropy.

The image data (MNIST and NotMNIST) used by the BNN is stored in `data`. Trained models are stored in `models/bnn/`. 

### Running the Code
Running the BNN code requires Python >= 3.6 with `pytorch`, `scipy` and `pyro` installed. Refer to the respective libraries websites for instructions on how to install them correctly. 

The code supports training and predicting on the GPU out of the box, and defaults to doing so if a GPU is found to be present on the system and the installed pytorch version has CUDA support. To prevent the network from being run on you GPU, set `use_cuda = False` in `settings.py`.

To train new models, first either delete the existing models, or change the model path in `settings.py`. Running `main.py` will then cause a new set of models to be trained, one for each training set size as defined in the paper. 

To see how the models perform on the MNIST and NotMNIST dataset, call `get_prediction_data` in `interface.py` with your choice of model training set size and data set (in the form of a string). `interface.py` contains a detailed description of the function's accepted parameter values. 

To use the trained models with a custom dataset, either call `get_prediction_data` with your dataset, or call one of the functions for prediction defined in `Net.py` with your instances (`prediction_data` requires both instances and targets).
