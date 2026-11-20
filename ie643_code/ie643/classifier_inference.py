import os
import argparse
import torch
from torchvision import transforms, models
from PIL import Image

# ==============================================================================
# 14 TARGET CLASSES (Must match training order exactly)
# ==============================================================================
TARGET_CLASSES = [
    "No Finding", "Lung Opacity", "Support Devices", "Pleural Effusion", 
    "Edema", "Pneumothorax", "Atelectasis", "Cardiomegaly", 
    "Fracture", "Consolidation", "Enlarged Cardiomediastinum", 
    "Lung Lesion", "Pneumonia", "Pleural Other"
]

# ==============================================================================
# MODEL INITIALIZATION
# ==============================================================================
def load_classifier(checkpoint_path, device):
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Dynamically grab the number of classes and class names if they exist in the checkpoint
    if 'config' in checkpoint and 'classes' in checkpoint['config']:
        class_names = checkpoint['config']['classes']
        num_classes = checkpoint['config']['num_classes']
    else:
        # Fallback to the default 14 if testing an older checkpoint
        class_names = TARGET_CLASSES 
        num_classes = 14

    # Initialize base architecture dynamically
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features, num_classes)
    
    # Load saved weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    return model, class_names

# ==============================================================================
# INFERENCE FUNCTION
# ==============================================================================
def predict_image(model, image_path, device, class_names): # Added class_names parameter
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
        
    predicted_idx = probabilities.argmax()
    predicted_class = class_names[predicted_idx] # Uses dynamic list
    predicted_prob = probabilities[predicted_idx]
    
    return predicted_class, predicted_prob, probabilities

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to the X-ray image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pth file")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and class names
    model, class_names = load_classifier(args.checkpoint, device)
    
    # Predict
    predicted_class, predicted_prob, all_probs = predict_image(model, args.image_path, device, class_names)
    
    # Print Results
    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    print(f"Image:       {os.path.basename(args.image_path)}")
    print(f"Prediction:  {predicted_class}")
    print(f"Probability: {predicted_prob * 100:.2f}%\n")
    
    print("All Probabilities:")
    print("-" * 50)
    # Sort and display all probabilities
    prob_dict = {class_names[i]: all_probs[i] for i in range(len(class_names))}
    for cls, prob in sorted(prob_dict.items(), key=lambda item: item[1], reverse=True):
        print(f"{cls:<30} {prob * 100:>6.2f}%")
        
if __name__ == "__main__":
    main()






# How to use the inference script
# Run the script from your terminal by pointing it to any X-ray image and the specific epoch checkpoint you want to evaluate.

# Bash
# python inference_classifier.py \
#     --image_path "CheXpert-v1.0-small/valid/patient64542/study1/view1_frontal.jpg" \
#     --checkpoint "real_checkpoint/epoch_10.pth"