# Image Captioning on the COCO Dataset
This project implements an image captioning system using deep learning techniques on the COCO (Common Objects in Context) dataset.

![Model Performance Comparison](model-comparison.svg)

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Exploration](#exploration)
4. [Data Augmentation](#data-augmentation)
5. [Models](#models)
6. [Results](#results)
7. [Trained Models](#trained-models)
8. [Failed and Alternative Experiments](#failed-and-alternative-experiments)
9. [Team](#team)

## Introduction
Image captioning is the process of generating a textual description for given images. This project explores various deep learning architectures to perform this task on the COCO dataset.

## Dataset
We use the COCO (Common Objects in Context) dataset:
- Large-scale object detection, segmentation, and captioning dataset from 2014
- Contains over 330,000 images with annotated objects
- Widely used in computer vision research and development
- Size: 17 GB

## Exploration
Our initial exploration revealed:
- Image heights are clustered around 500 pixels
- Image widths are clustered around 600 pixels
- Captions contain between 25 to 75 words with a peak around 50 words

## Data Augmentation
We implemented two levels of data augmentation:
1. Base Augmentation:
   - Random horizontal flip (50% chance)
   - Random rotation (±0.2 radians or ±11.5°)
   - Random contrast adjustment (±0.3 factor)
2. Improved Augmentation:
   - All base augmentations
   - Random zoom (±20% of original size)
   - Random brightness adjustment (±20% intensity)

## Models
We experimented with several model architectures:
1. LSTM with Custom CNN
2. Transformer with Custom CNN
3. Transformer with EfficientNet CNN (B0 and B7)
4. Variations of Transformer architecture:
   - With Leaky ReLU
   - With 2x CNN Size
   - With +2 Conv layers
   - With 2x num_heads

## Results
- Transformer-based models outperformed LSTM-based models
- EfficientNet (both B0 and B7) outperformed custom CNN architectures
- EfficientNetB0 achieved the best performance considering the trade-off between accuracy and training time

## Trained Models
The trained models can be found in the following Google Drive folder:
[https://drive.google.com/drive/u/1/folders/1kc_KL3p7331Nzg-Slmc5PbQ6wiJtlZYP](https://drive.google.com/drive/u/1/folders/1kc_KL3p7331Nzg-Slmc5PbQ6wiJtlZYP)
Please refer to this folder for the latest versions of our trained models.

## Failed and Alternative Experiments
1. Florence-2 Finetuning: The predicted captions were not semantically correct.
2. Adding more transformer layers: This could potentially enhance model complexity with more resources.
3. Evaluating other architectures (e.g., GRU): Exploring alternative architectures might improve performance.

## Team
- Antonio Sabbatella (Data Science Student)
- Alex Calabrese (Data Science Student)

University of Milano-Bicocca
