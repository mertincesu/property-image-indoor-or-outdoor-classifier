---
license: mit
tags:
  - image-classification
  - computer-vision
  - binary-classifier
model-details:
  architecture: MobileNetV2
  classes: ["Indoor", "Outdoor"]
  accuracy: 96%
  input-size: "160x160 RGB"
---

# Indoor vs Outdoor Classifier

A binary image classifier that determines whether a property image is **indoor** or **outdoor** with **96% accuracy**.

## Model Details

- **Model Type:** MobileNetV2-based binary classifier  
- **Classes:** Indoor, Outdoor  
- **Accuracy:** 96% on validation set  
- **Input Size:** 160x160 RGB images  

## Training Data

The model was trained on **2,000 curated Airbnb property images**, split evenly between indoor and outdoor scenes. The dataset was manually verified to ensure high-quality training examples.

## Usage

### Install Dependencies

pip install transformers pillow