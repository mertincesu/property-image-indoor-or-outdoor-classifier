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

  import requests
  import torch
  from torchvision import transforms, models
  import torch.nn as nn
  from PIL import Image
  import io

  def classify_image_huggingface(image_path, repo_id="mertincesu/property-indoor-or-outdoor-classifier"):
      
      # Define model structure
      model = models.mobilenet_v2(weights=None)
      num_ftrs = model.classifier[1].in_features
      model.classifier[1] = nn.Linear(num_ftrs, 2)
      
      # Load model weights directly from Hugging Face
      model_url = f"https://huggingface.co/{repo_id}/resolve/main/pytorch_model.bin"
      
      try:
          # Download the model file
          print(f"Downloading model from {model_url}")
          response = requests.get(model_url)
          response.raise_for_status()
          
          # Load the model weights
          model_binary = io.BytesIO(response.content)
          model.load_state_dict(torch.load(model_binary, map_location=device))
          model.to(device)
          model.eval()
          
          # Define image transformation
          transform = transforms.Compose([
              transforms.Resize((160, 160)),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
          ])
        
          # Load and process the image
          img = Image.open(image_path).convert('RGB')
          img_tensor = transform(img).unsqueeze(0).to(device)
          
          # Make prediction
          with torch.no_grad():
              outputs = model(img_tensor)
              probs = torch.nn.functional.softmax(outputs, dim=1)
              _, predicted = torch.max(outputs, 1)
          
          classes = ['indoor', 'outdoor']
          result = {
              'class': classes[predicted.item()],
              'confidence': probs[0][predicted.item()].item() * 100
          }
          
          print(f"Class: {result['class']}")
          print(f"Confidence: {result['confidence']:.2f}%")
          return result
          
      except Exception as e:
          print(f"Error: {e}")
          return None
  
  if __name__ == "__main__":
      import sys
      
      if len(sys.argv) > 1:
          image_path = sys.argv[1]
      else:
          image_path = input("Enter path to image: ")
      
      classify_image_huggingface(image_path)