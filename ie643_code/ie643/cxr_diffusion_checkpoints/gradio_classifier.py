import gradio as gr
import torch
import os
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

# ==============================================================================
# --- CONFIGURATION ---
# Update these paths to point to your best checkpoint files!
# ==========================================
REAL_MODEL_PATH = "real_classifier_checkpoints/best_model.pth"
SYNTHETIC_MODEL_PATH = "real_synthetic_classifier_checkpoints_prism/epoch_9.pth"

TARGET_CLASSES = [
    "No Finding", "Lung Opacity", "Support Devices", "Pleural Effusion", 
    "Edema", "Pneumothorax", "Atelectasis", "Cardiomegaly", 
    "Fracture", "Consolidation", "Enlarged Cardiomediastinum", 
    "Lung Lesion", "Pneumonia", "Pleural Other"
]

# Global variables for caching to prevent reloading every time you click predict
current_model_type = None
model_cache = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# MODEL LOADING
# ==============================================================================
def get_densenet121_classifier(num_classes=14):
    """Initializes the exact architecture used during training."""
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    # Must match the training script exactly to avoid state_dict errors
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    return model

def load_model(model_choice):
    global current_model_type, model_cache
    
    # Return cached model if it's already loaded
    if current_model_type == model_choice and model_cache is not None:
        return model_cache

    print(f"\nSwitching to: {model_choice}")
    
    # Determine which path to load
    if model_choice == "Real Data Model":
        ckpt_path = REAL_MODEL_PATH
    else:
        ckpt_path = SYNTHETIC_MODEL_PATH
        
    if not os.path.exists(ckpt_path):
        raise gr.Error(f"Checkpoint not found at: {ckpt_path}. Please update the configuration paths at the top of the script.")

    # Initialize and load weights
    print(f"Loading weights from {ckpt_path}...")
    model = get_densenet121_classifier(num_classes=len(TARGET_CLASSES))
    
    # weights_only=False added to prevent the PyTorch 2.6 unpickling error
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    current_model_type = model_choice
    model_cache = model
    print("Model loaded successfully!")
    
    return model

# ==============================================================================
# INFERENCE LOGIC
# ==============================================================================
def predict_xray(image, model_choice):
    if image is None:
        raise gr.Error("Please upload an image first!")
        
    try:
        # Load correct model
        model = load_model(model_choice)
        
        # Image Preprocessing (Matches your training DataLoader)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Make sure image is RGB
        image = image.convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            # Use Sigmoid if you trained with BCE, but since you used CrossEntropyLoss, we use Softmax
            probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
            
        # Format for Gradio's Label component (Dictionary of {Class: Probability})
        # We convert to float to ensure Gradio JSON serializes it properly
        result_dict = {TARGET_CLASSES[i]: float(probabilities[i]) for i in range(len(TARGET_CLASSES))}
        
        return result_dict
        
    except Exception as e:
        raise gr.Error(str(e))

# ==============================================================================
# GRADIO UI
# ==============================================================================
with gr.Blocks(title="CXR Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🩻 Chest X-Ray Pathology Classifier
        Upload a Chest X-ray and compare how models trained on purely real data versus mixed synthetic data perform.
        """
    )
    
    with gr.Row():
        # LEFT COLUMN (Inputs)
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=["Real Data Model", "Real + Synthetic Data Model"],
                value="Real Data Model",
                label="1. Select Classifier Model",
                info="Choose which trained weights to use for inference."
            )
            
            image_input = gr.Image(
                type="pil", 
                label="2. Upload Chest X-Ray",
                image_mode="RGB"
            )
            
            predict_btn = gr.Button("🔍 Analyze X-Ray", variant="primary", size="lg")
            
        # RIGHT COLUMN (Outputs)
        with gr.Column(scale=1):
            # The gr.Label component automatically creates beautiful progress bars for probabilities
            output_label = gr.Label(
                num_top_classes=5, # Only shows the top 5 highest probabilities to keep UI clean
                label="Predicted Pathologies"
            )
            
    # Connect the button
    predict_btn.click(
        fn=predict_xray,
        inputs=[image_input, model_dropdown],
        outputs=[output_label]
    )

if __name__ == "__main__":
    # Launch app (share=True provides a public link)
    demo.launch(share=True, debug=True)