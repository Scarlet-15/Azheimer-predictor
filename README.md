# Azheimer-predictor
Deep Learning Neural Net for HAXIOS-2023
Streamlit App Link: 
[click here](https://hwaseem04-urban-rural-classification-app-urj25b.streamlit.app/)

## Introduction
Alzheimer's disease is a type of brain disease caused by damage to nerve cells (neurons) in the brain.
It is a progressive disease beginning with mild memory loss and possibly leading to loss of the ability to carry on a conversation and respond to the environment. Alzheimer's disease involves parts of the brain that control thought, memory, and language.The neurons damaged first are those in parts of the brain responsible for memory, language and thinking. As a result, the first symptoms of Alzheimer’s disease tend to be memory, language and thinking problems.

## How we have approached
We have used Convolutional Neural Network to train our Neural Network. To achieve maximum accuracy, we have implemented transfer learning. Over 5000 images of 4 classes are used to train the model. The classes are
* Mild Demented
* Moderate Demented
* Non Demented
* Very Mild Demented
We have downloaded the training data from Kaggle [click_here](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images).

## How to use
* Refer to our video demo [in youtube](https://youtu.be/vcRnDZyhBuY).
![demo video](demo.gif)
* You can use data that I have in this folder by cloning this repo, or you can manually use data from internet.

* **Test with data that are in similar grounds of training data**.
* **Upload only MRI scanned images**.

## Implementation
* Picked up `VGG-16` Architecture for transfer learning.
* You can refer to our [Jupyter file](neuraltraining.ipynb) for implementation of transfer learning.

## Improvements
* Incase certain images are misclassified please raise an issue along with the image.
* Since the hackathon's duration is 36hours, the model predicts almost every case of Azheimer. However the model can be finetuned to perfection if the training time is increased with a better GPU.
