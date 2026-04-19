[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)


# **Spiking Neural Network for DVS128 Gesture Recognition**

This repository contains the source code for the paper "Fully Integrated Memristive Spiking Neural Network with Analog Neurons for High-Speed Event-Based Data Processing". This code demonstrates a Spiking Neural Network (SNN) implemented with PyTorch and the [`SLAYER`](https://github.com/bamsumit/slayerPytorch) library. The network is trained and tested on the DVS128 Gesture dataset to perform motion recognition.

## Contents
- [Overview](#spiking-neural-network-for-dvs128-gesture-recognition)
- [Description](#description)
- [File Structure](#file-structure)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Usage](#usage)
- [Model Architecture](#model-architecture)

## **Description**

The core of the project is `motion_recognition.py`, which defines:

* A PyTorch Dataset class (`DvsGestureDataset`) to load and preprocess the DVS128 Gesture data (`.mat` files).  
* A `Network` class defining the SNN architecture using SlayerSNN layers (`snn.layer`, `snn.dense`, `snn.pool`). The network consists of pooling and fully connected layers.  
* Function for plotting the confusion matrix to visualize prediction accuracy.  
* A main execution block that loads a pre-trained model (`model.tar`), loads the test dataset, performs inference, calculates accuracy, and plots the confusion matrix.

The network configuration, including simulation parameters, neuron properties, and training/testing data paths, is defined in `network.yaml`.

## **File Structure**

* **`motion_recognition.py`**: Main script defining the SNN model, data loading, and evaluation logic.  
* **`network.yaml`**: Configuration file for simulation, neuron, and dataset parameters. 
* **`model.tar`**: Contains the pre-trained model weights.  
* [**`DvsGesture.zip`**](https://drive.google.com/file/d/1CGMGjbJekY8cenA6CUTOx6XSolWYH0Gx/view?usp=sharing): Contains the DVS128 Gesture dataset (`.mat` files) and sample lists (`.txt` files).

## **Requirements**

* Python  
* PyTorch  
* NumPy  
* SciPy
* Matplotlib
* SlayerSNN

You can typically install these using `pip`:
```bash
pip install numpy scipy matplotlib
```

The installation method of SLAYER library (`slayerSNN`) is available [here](https://github.com/bamsumit/slayerPytorch). 

The main script `motion_recognition.py` has been tested with Python 3.12 and CUDA 12.4.

## **Dataset**

* **Download**: [**`DvsGesture.zip`**](https://drive.google.com/file/d/1CGMGjbJekY8cenA6CUTOx6XSolWYH0Gx/view?usp=sharing)
* The code uses the DVS128 Gesture dataset. The version provided in this repository is directly derived from the original [IBM DVS128 Gesture dataset](http://research.ibm.com/dvsgesture/) and is generated at a 5-ms temporal resolution.
* The paths to the training and testing data directories and sample files need to be specified in `network.yaml` under the `training:path` section:  
  * `inTrain`: Path to the training data directory.  
  * `inTest`: Path to the testing data directory.  
  * `train`: Path to the file listing training samples.  
  * `test`: Path to the file listing test samples.  
* The dataset files are in `.mat` format and contain spike event data. A sample with size \[2, 128, 128, 300\] can be checked in MATLAB using the code:
  ```Matlab
  load('00001.mat')  
  implay(squeeze(Voxel(1, :, :, :) * 255))
  ```
## **Configuration**

Network and simulation parameters are controlled via `network.yaml`:

* **simulation**: Time step (`Ts`) and total simulation time (`tSample`).  
* **neuron**: SNN neuron model parameters (type, threshold, time constants, etc.).  
* **training**: Error calculation settings and dataset paths.

## **Usage**

1. **Prepare the Dataset:** Ensure the DVS128 Gesture dataset (included in the repository) is structured according to the paths specified in `network.yaml`.  
2. **Configuration:** Verify and update `network.yaml` with the correct paths and desired parameters.  
3. **Pre-trained Model:** Ensure the pre-trained model file (`model.tar`) is available in the same directory or update the path in `motion_recognition.py`.  
4. **Run Inference & Evaluation:** Execute the Python script:
   ```Bash
   python motion_recognition.py
   ```
   This will load the pre-trained model `model.tar` and the test data, run the network, print testing statistics (loss, accuracy), and display a confusion matrix showing the classification performance.

## **Model Architecture**

The SNN architecture defined in the `Network` class consists of:

1. Input Padding (`F.pad`)  
2. Pooling Layer (`slayer.pool`)  
3. Fully Connected Layer 1 (`slayer.dense`) with weight clamping  
4. Spike Generation (`slayer.spike`, `slayer.psp`)  
5. Fully Connected Layer 2 (`slayer.dense`) with weight clamping  
6. Spike Generation (`slayer.spike`, `slayer.psp`)  
7. Output Fully Connected Layer (`slayer.dense`) with weight clamping  
8. Output Spike Generation (`slayer.spike`, `slayer.psp`)

The neuron model used is SRM, configured in `network.yaml`.
