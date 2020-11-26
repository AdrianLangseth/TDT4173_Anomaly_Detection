# TDT4173_Anomaly_Detection

This repository contains code for the 2020 TDT4173 project task. The chosen topic is probabilistic learning, and the project focuses on comparing the attributes and performance of standard feed forward neural networks (with and without dropout) and those of a Bayesian neural network. The project's aim is not to create the best possible networks/models, but to create networks that are easily compared.

The project source consists of three submodules:
* bnn - Source code for the BNN.
* ffnn - Source code for the FFNN. Some visualization code can also be found here.
* visualization - Source code for the visualizations presented in the paper.

Due to the stochastic nature nature of this work (dropout, training & validation set contents are randomized, BNNs are inherently probabilistic), the results presented in the paper may not be exactly reproducible. The general trends that we discuss however, will show up in any retrained models.

## Sources

### Vizualization

Dennis T, 2019. _Confusion Matrix Visualization_. https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea (accessed 31.10.2020)

NTNU AI Labs, 2020. _tdt7143-2020_. https://github.com/ntnu-ai-lab/tdt4173-2020 (accessed 31.10.2020)


## Feed Forward Neural Network (FFNN) w/o Dropout

A simple feed-forward neural network type which classifies MNIST images. The network consists of one input layer with 784 (28*28) nodes, one hidden layer with 512 nodes, and an output layer with 10 nodes.
In addition to this, one type of network implemented is the exact same with the variation of having a dropout layer on the hidden layer nodes.

The ffnn and dropout models trained on a training set of 50 000 images and a maximum of 1000 epochs of training, with early stopping implemented, have both over 98% accuracy.

### Project Structure
Code relating to the FFNN can be found in `src/ffnn/`. The folder contains the following files:
* `data_load.py` Contains code for loading and processing the MNIST and NotMNIST datasets.
* `FFNN_predictor.py` Contains code for all prediction and entropy calculation.
* `hyp_opt.py` Code for hyperparameter optimization.  
* `mnist.npz` npz-file for the MNIST image dataset.
* `Multiple_FFNN_Dropout.py` Contains code for creating and saving the models which employ dropout. in reality it is a convenience wrapper on the building function in `Multiple_FFNN.py`.
* `Multiple_FFNN.py`  Code for creating and saving FFNN models without the dropout layers. All model building happens here.
* `notMNIST.py` Code for testing models on the notMNIST dataset, and calculating entropy.
* `visualization_interface.py` Interface to the FFNN for the visualization of its metrics and predictions, using convenience wrappers to show only the need-to-know for the developer responsible for visualization.
* `Visualization.py` Code for generating the visualizations used in the report.
* `webcode.py` Contains a small snippet of code for retrieving the BNN's prediction entropy for the 10 instances with highest FFNN prediction entropy.

The image data (MNIST and NotMNIST) used by the model is stored in `mnist.npz` and the `notMNIST_all` folder, respectively. The trained models are stored in the folders corresponding to their type.

### Running the Code
Running the FFNN code requires Python >= 3.6 with `tensorflow == 2.3.1`, `hyperas`, `hyperopt >= 0.2.5` and `scipy >= 1.5.4` installed. `hyperas` and `hyperopt` are only necessary if one is to perform hyperparameter optimization. Refer to the respective libraries websites for instructions on how to install them correctly.

To train new models, first either delete the existing models. Running `Multiple_FFNN.py` or `Multiple_FFNN_Dropout.py` will then train a new set of models, one for each training set size as defined in the paper.

To see how the models perform on the MNIST and notMNIST dataset, use the functions in `FFNN_predictor.py`. Running the file itself will test the models against the test set and output their accuracies. **Note:** This will take a while, be patient. The models are making 2.02·10^7 predictions. It will push out the results as they come in, but it might take around 10 minutes for all results to be generated, depending on the hardware it runs on. On slower hardware, this might be higher. The dropout models are responsible for 99% of the predictions, which means a possible remedy in such situations would be to change the variables `dropout_runs` in the prediction functions. It is now 100, which means for each image, it would be predicted 100 times. Lowering it to 10 would give reduce the runtime by up to 89%, however the models would correspondingly be less like ta Bayes approximation and more alike the regular FFNN. 

Efforts have not been made to run the code on the GPU, as it runs acceptably on the CPU and has therefore not been prioritized due to the time constraint of the project.

To use the trained models with a custom dataset, employ the `model_predictor` in `FFNN_predictor.py` with the correct parameters.


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
