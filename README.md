# **Speech Emotion Recognition (SER)**

This project implements a Speech Emotion Recognition (SER) system using a Convolutional Recurrent Neural Network (CRNN). The model is trained on the [EmoDB dataset](https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb) to classify emotions from speech audio.

## Table of Contents

1. [Installing Libraries](#1-installing-libraries)
2. [Preprocessing and Creating the Labeled Dataset](#2-preprocessing-and-creating-the-labeled-dataset)
3. [Building and Training the Model](#3-building-and-training-the-model)
4. [Visualizing the Trained Model’s Performance](#4-visualizing-the-trained-models-performance)
5. [Confusion Matrix and Classification Report of Pre-Trained Model](#5-confusion-matrix-and-classification-report-of-pre-trained-model)



## **1\. Installing Libraries**

Ensure you have the necessary libraries installed for this project. Each library plays a crucial role in the preprocessing, model building, and evaluation stages:

* **Librosa**: Used for audio processing and feature extraction (e.g., loading audio files, resampling, and generating Mel-spectrograms).  
  * Imported Modules: `librosa`, `librosa.effects`, `librosa.feature`  
* **TensorFlow**: Utilized to build, train, and load the Convolutional Recurrent Neural Network (CRNN) model.  
  * Imported Modules: `tensorflow`, `tensorflow.keras.models`, `tensorflow.keras.layers`, `tensorflow.keras.regularizers`, `tensorflow.keras.optimizers`  
* **Pandas**: Handles data organization and management, particularly for datasets.  
  * Imported Modules: `pandas`  
* **Seaborn**: Provides functionality for visualizing results, such as plotting the confusion matrix as a heatmap.  
  * Imported Modules: `seaborn`  
* **Scipy**: Specifically used for the Wiener filter, which helps reduce noise in audio files.  
  * Imported Modules: `scipy.signal`  
* **NumPy**: Handles numerical computations and matrix operations, essential for data manipulation.  
  * Imported Modules: `numpy`  
* **Scikit-learn**: Used for encoding categorical labels and splitting datasets into training and testing sets.  
  * Imported Modules: `sklearn.preprocessing.OneHotEncoder`, `sklearn.model_selection.train_test_split`  
* **Matplotlib**: For visualizing data, including plotting confusion matrices.  
* **Seaborn**: Enhances data visualization, especially for creating heatmaps such as confusion matrix plots.

Additionally, `os` is used for file and directory handling throughout the project.

## **2\. Preprocessing and Creating the Labeled Dataset**

The preprocessing of audio files involves several key steps, each using specific modules:

* **Resampling**: The audio is resampled to 44.1 kHz for standardization using `librosa.resample`.  
* **Silence Trimming**: Silence from the beginning and end of the audio is removed with `librosa.effects.trim`.  
* **Noise Reduction**: Background noise is reduced by applying Wiener filtering with `scipy.signal.wiener`.  
* **Zero Padding/Truncation**: Audio files are padded or truncated to a fixed duration of 3 seconds using `numpy.pad`.  
* **Mel-Spectrogram Generation**: The waveform is converted into a Mel-spectrogram using `librosa.feature.melspectrogram`.  
* **Log Mel-Spectrogram Conversion**: The Mel-spectrogram is then converted to a log scale using `librosa.power_to_db`.

These steps ensure the audio is normalized, cleaned, and ready for input into the deep learning model.

## **3\. Building and Training the Model**

The model is a **Convolutional Recurrent Neural Network (CRNN)**, designed to capture both spatial and temporal features of audio data. It consists of:

* **Input Layer**: Takes in 128x259 mel-spectrograms with 1 channel.  
* **Convolutional Layers**: Four Conv2D layers with increasing filter sizes (64, 64, 128, 128), followed by batch normalization and max pooling. These layers extract spatial features from the spectrogram.  
* **Reshape Layer**: The 2D output from the convolutional layers is reshaped into a 3D tensor to be passed into the LSTM layers.  
* **Bidirectional LSTM**: A single BiLSTM layer with 128 units captures temporal dependencies in the audio data, learning how features change over time.  
* **Dense Layer**: The final dense layer uses softmax activation to classify the input into 8 possible emotion categories.

The model is compiled using the **Adam optimizer** with a learning rate of 0.0001, and the **Categorical Crossentropy** loss function. The model is trained with a batch size of 16 for 10 epochs, using accuracy as the evaluation metric.

## **4\. Visualizing the Trained Model’s Performance**

If you decide to train your own model following the steps in this notebook, here’s how you can visualize its performance:

1. **Run the testing cells**: After training your model, proceed to the section where the model is evaluated on the test data.  
2. **Generate confusion matrix**: The confusion matrix will show how well the model classified emotions, and can be visualized as a heatmap.  
3. **Display classification report**: The classification report will show precision, recall, and F1-score for each emotion class.

The visualization steps are already provided in the notebook for easy use.

## **5\. Confusion Matrix and Classification Report of Pre-Trained Model**


Below are the performance results of the pre-trained CRNN model I developed for speech emotion recognition:

### **Confusion Matrix**

This heatmap shows the normalized confusion matrix, representing how well the model classifies emotions. Each row corresponds to the true labels, and each column corresponds to the predicted labels. The values indicate the proportion of correct and incorrect classifications.

![normalized confusion matrix](https://github.com/user-attachments/assets/df951436-5181-4ec9-a288-9343358faddf)

### **Classification Report**

The classification report provides detailed performance metrics such as **precision**, **recall**, and **F1-score** for each emotion class. This helps in understanding how the model performs on each emotion category.

![classification reprt](https://github.com/user-attachments/assets/20c129bb-806b-4ab4-8568-4f3ac60fd1d5)

