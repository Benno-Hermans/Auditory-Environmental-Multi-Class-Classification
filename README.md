# Auditory-Environmental-Multi-Class-Classification

## Overview
This project contains the notebook and supporting code for a custom Convolutional Neural Network (CNN) trained from scratch to classify environmental audio into 50 distinct sound categories using the ESC-50 dataset. Raw audio clips are converted to log scaled mel spectrograms and fed into an 8 layer CNN pipeline built and trained in Google Colab.

The ESC-50 dataset was sourced from Hugging Face (user: ashraq) and contains 2,000 five second recordings across 50 classes including animal sounds, natural soundscapes, human non-speech sounds, domestic sounds, and urban noise.

## Motivation
Environmental sound classification has practical applications in wildlife monitoring, assistive hearing technology, smart home systems, and industrial fault detection. This project explores how far a purpose-built CNN trained from scratch can go on a small, challenging dataset and without relying on large pretrained models through the combination of careful audio preprocessing, regularization, and data augmentation.

## Features
- Full audio-to-spectrogram preprocessing pipeline (STFT → mel filterbank → log scaling → padding)
- Custom 8 layer CNN with batch normalization, dropout, and global average pooling
- 5-Fold cross validation using ESC-50's predefined folds
- Data augmentation including noise, gain shifts, time shifts, frequency masking, and time masking
- Ablation study comparing model performance with and without augmentation
- Evaluation via accuracy, F1 score, confusion matrix, and training curves

## How to Run
1. Open the notebook (EEL4810_AudioClassification.ipynb) in Google Colab
2. Make sure you're using a GPU runtime (Runtime → Change runtime type → T4 GPU)
3. Run all cells top to bottom. The notebook will install dependencies, download the dataset via Hugging Face, preprocess audio, train the model across all 5 folds, and generate evaluation outputs automatically
4. For new, full training iterations, run all cells.
5. To only evaluate the pre-trained models, import the datasets, run cells that set up the CNN model and datasets, and then run the final notebook cell.

## *Files
- EEL4810_AudioClassification.ipynb → Main Notebook File
- before_augmentation.pth → Saved Model Before Data Augmentation
- after_augmentation.pth → Saved Model After Data Augmentation
*The BEATs .pth file utilized as a baseline comparison is too large to be hosted on github. The download for an array of these models can be found here: https://github.com/microsoft/unilm/tree/master/beats


## Authors
- Carter Harman
- Benno Hermans
- Zackary Groth
- Giovanni DeAngeles
