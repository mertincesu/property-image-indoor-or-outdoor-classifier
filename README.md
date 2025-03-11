---
license: mit
tags:
  - image-classification
  - indoor-outdoor
  - real-estate
  - mobilenetv2
  - property-classifier
  - property-classification
  - binary-classifier
datasets:
  - airbnb-property-images
pipeline_tag: image-classification
widget:
  - src: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/indoor-example.jpg
    example_title: Indoor Example
  - src: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/outdoor-example.jpg
    example_title: Outdoor Example
---

# Indoor vs Outdoor Classifier

A binary image classifier that determines whether a property image is indoor or outdoor with 96% accuracy.

## Model Details

- **Model Type**: MobileNetV2-based binary classifier
- **Classes**: Indoor, Outdoor
- **Accuracy**: 96% on validation set
- **Input Size**: 160x160 RGB images

## Training Data

The model was trained on 2,000 curated Airbnb property images, split evenly between indoor and outdoor scenes. The dataset was manually verified to ensure high-quality training examples.

## Use Cases

- Real estate listing automation
- Property image organization
- Virtual tour preparation
- Interior design vs. architecture applications

## License

This model is released under MIT License.

## Code Implementation

Example usage code has been provided in example_usage.py under model files.