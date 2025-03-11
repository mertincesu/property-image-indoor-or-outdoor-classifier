import requests
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import io

def classify_image_huggingface(image_path, repo_id="mertincesu/property-indoor-or-outdoor-classifier"):
    # Automatically select the best available device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                           "mps" if torch.backends.mps.is_available() else 
                           "cpu")
    print(f"Using device: {device}")

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