# FER2013 Facial Expression Recognition

## Overview
This project implements a Facial Expression Recognition (FER) model using the FER2013 dataset. The model is built with PyTorch and utilizes a Convolutional Neural Network (CNN) to classify facial expressions into seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Features
- Preprocessing and loading of FER2013 dataset.
- Custom CNN architecture for facial expression classification.
- Training and evaluation pipeline with accuracy and loss tracking.
- Learning rate scheduling and optimization options.
- Precision, recall, F1-score, and confusion matrix evaluation.
- Data visualization utilities to inspect dataset samples.

## Installation
To set up the project, ensure you have Python installed and run the following commands:

```sh
pip install torch torchvision matplotlib scikit-learn tqdm
```

## Project Structure

```
FER2013-Facial-Expression-Recognition/
│── data/fer2013/
│   ├── icml_face_data.csv
│   ├── train.csv
│   ├── test.csv
│── infra/
│   ├── DataProcessor.py   # Dataset processing and dataloader utilities
│── models/
│   ├── FER_Model.py       # CNN model definition
│── Trainer.py             # Training and evaluation logic
│── main.py                # Main execution script
│── README.md              # Project documentation
```
Note: Download the datasets (.csv files) from https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=train.csv and save to data/fer2013/

## Usage
Run the following command to train and evaluate the model:
```sh
python main.py
```

The script will:
1. Load and preprocess the FER2013 dataset.
2. Train the CNN model on the dataset.
3. Evaluate the model on the test dataset.
4. Display accuracy, precision, recall, F1-score, and confusion matrix.
5. Plot training and validation loss/accuracy trends.

## Model Architecture
The CNN model consists of:
- Multiple convolutional layers with batch normalization and ReLU activations.
- Max-pooling layers to reduce spatial dimensions.
- Fully connected layers for classification.
- Cross-entropy loss function with an SGD optimizer.

## Results & Evaluation
- Training progress is logged at regular intervals.
- Evaluation metrics include:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix

## Dataset
The FER2013 dataset consists of grayscale facial images (48x48 pixels) labeled with one of seven emotions. It is automatically downloaded when running the script.
