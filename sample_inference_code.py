
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

def classify_image(image_path, model_path="indoor_outdoor_classifier_from_scratch/best_model.pth"):
    # Check if MPS (Apple Silicon GPU) is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define the model architecture
    def get_model():
        model = models.mobilenet_v2(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)
        return model
    
    # Load the model
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Prepare image transformation
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        probabilities = torch.nn.functional.softmax(output, dim=1)
    
    # Get class label and probability
    class_names = ['indoor', 'outdoor']
    predicted_class = class_names[predicted.item()]
    probability = probabilities[0][predicted.item()].item()
    
    return {
        'class': predicted_class,
        'probability': probability * 100,
        'all_probabilities': {
            class_names[i]: probabilities[0][i].item() * 100 for i in range(len(class_names))
        }
    }

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter the path to the image: ")
    
    result = classify_image(image_path)
    print(f"Class: {result['class']} ({result['probability']:.2f}%)")
