# TDT4173_Anomaly_Detection

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

To see how the models perform on the MNIST and notMNIST dataset, use the functions in `FFNN_predictor.py`. Running the file itself will test the models against the test set and output their accuracies. **Note:** This will take a while, be patient. The models are making 2.02*10^7 predictions. It will push out the results as they come in, but it might take around 10 minutes for all results to be generated, depending on the hardware it runs on. Efforts have not been made to run the code on the GPU, as it runs acceptably on the CPU and has therefore not been prioritized due to the time constraint of the project.

To use the trained models with a custom dataset, employ the `model_predictor` in `FFNN_predictor.py` with the correct parameters.
