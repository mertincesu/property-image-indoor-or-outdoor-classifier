---
license: mit
---
Indoor vs Outdoor Classifier
A binary image classifier that determines whether a property image is indoor or outdoor with 96% accuracy.
Model Details

Model Type: MobileNetV2-based binary classifier
Classes: Indoor, Outdoor
Accuracy: 96% on validation set
Input Size: 160x160 RGB images

Training Data
The model was trained on 2,000 curated Airbnb property images, split evenly between indoor and outdoor scenes. The dataset was manually verified to ensure high-quality training examples.
Usage
pythonCopyfrom transformers import pipeline
from PIL import Image

# Load model
classifier = pipeline("image-classification", model="yourusername/indoor-outdoor-classifier")

# Classify an image
image = Image.open("path/to/your/image.jpg")
result = classifier(image)
print(f"Class: {result[0]['label']}, Confidence: {result[0]['score']:.2%}")
Use Cases

Real estate listing automation
Property image organization
Virtual tour preparation
Interior design vs. architecture applications

License
This model is released under MIT License.