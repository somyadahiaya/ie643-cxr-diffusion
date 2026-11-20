import os
import argparse
from xml.parsers.expat import model
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import json
import random

# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==============================================================================
# DATASET CLASS
# ==============================================================================
class CheXpertClassifierDataset(Dataset):
    def __init__(self, csv_path, base_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.base_dir = base_dir
        self.transform = transform
        
        self.target_classes = [
            "No Finding", "Lung Opacity", "Support Devices", "Pleural Effusion", 
            "Edema", "Pneumothorax", "Atelectasis", "Cardiomegaly", 
            "Fracture", "Consolidation", "Enlarged Cardiomediastinum", 
            "Lung Lesion", "Pneumonia", "Pleural Other"
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_path = os.path.normpath(os.path.join(self.base_dir, row['Path']))
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label_vector = row[self.target_classes].values.astype(float)
        label_idx = int(label_vector.argmax())
        
        return image, torch.tensor(label_idx, dtype=torch.long)

# ==============================================================================
# DATALOADER FACTORY
# ==============================================================================
def get_classifier_dataloaders(train_csv, test_csv, base_dir, batch_size=32, num_workers=1):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Added slight zoom
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Uncommented this
        transforms.ToTensor(),
        normalize
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    train_dataset = CheXpertClassifierDataset(train_csv, base_dir, transform=train_transform)
    test_dataset = CheXpertClassifierDataset(test_csv, base_dir, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_dataset, train_loader, test_loader

# ==============================================================================
# DENSENET121 MODEL
# ==============================================================================
def get_densenet121_classifier(num_classes=14):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    in_features = model.classifier.in_features
    # model.classifier = nn.Linear(in_features, num_classes)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    return model
# def get_densenet121_classifier(num_classes=14):
#     model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    
#     # YOU MUST ADD THIS LOOP TO FREEZE THE BASE LAYERS
#     for param in model.parameters():
#         param.requires_grad = False
        
#     in_features = model.classifier.in_features
#     model.classifier = nn.Sequential(
#         nn.Dropout(p=0.2),  # Added dropout for regularization
#         nn.Linear(in_features, num_classes)
#     )
#     return model

# ==============================================================================
# EVALUATION FUNCTION
# ==============================================================================
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, targets in tqdm(
            dataloader,
            desc="Evaluating",
            leave=False
        ):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # Overall metrics
    acc = accuracy_score(all_targets, all_preds)

    precision = precision_score(
        all_targets,
        all_preds,
        average='macro',
        zero_division=0
    )

    recall = recall_score(
        all_targets,
        all_preds,
        average='macro',
        zero_division=0
    )

    f1 = f1_score(
        all_targets,
        all_preds,
        average='macro',
        zero_division=0
    )

    # Per-class metrics
    per_class_report = classification_report(
        all_targets,
        all_preds,
        target_names=dataloader.dataset.target_classes,
        output_dict=True,
        zero_division=0
    )

    # Confusion matrix
    conf_matrix = confusion_matrix(
        all_targets,
        all_preds,
        labels=list(
            range(len(dataloader.dataset.target_classes))
        )
    )

    # AUROC
    try:
        auroc = roc_auc_score(
            all_targets,
            all_probs,
            multi_class='ovr',
            average='macro'
        )
    except ValueError:
        auroc = float('nan')

    return (
        avg_loss,
        acc,
        precision,
        recall,
        f1,
        auroc,
        per_class_report,
        conf_matrix
    )
# ==============================================================================
# MAIN TRAINING SCRIPT
# ==============================================================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--base_dir", type=str, default=os.getcwd())
    parser.add_argument("--output_dir", type=str, default="classifier_checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a specific .pth file to resume training"
    )

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    train_dataset, train_loader, test_loader = get_classifier_dataloaders(
        args.train_csv, args.test_csv, args.base_dir, batch_size=args.batch_size
    )

    # 2. Init Model, Loss, Optimizer
    model = get_densenet121_classifier(num_classes=14).to(device)

    # label_counts = train_dataset.df[train_dataset.target_classes].sum(axis=0).values
    # total_samples = label_counts.sum()
    # class_weights = total_samples / (len(train_dataset.target_classes) * (label_counts + 1e-5))
    # class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    # criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    label_counts = train_dataset.df[train_dataset.target_classes].sum(axis=0).values
    total_samples = label_counts.sum()
    standard_weights = total_samples / (len(train_dataset.target_classes) * (label_counts + 1e-5))
    smoothed_weights = np.sqrt(standard_weights)
    class_weights_tensor = torch.tensor(smoothed_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    # optimizer = optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    start_epoch = 0
    history_log = []

    # 3. Resume Logic
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming training from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
        # Load existing log history if available
        log_path = os.path.join(args.output_dir, "log.json")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                history_log = json.load(f)

    # 4. Training Loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, targets in progress_bar:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate train accuracy on the fly
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)
            
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = correct_train / total_train
        
        # 5. Evaluate on Test Set
        (
            val_loss,
            val_acc,
            val_prec,
            val_rec,
            val_f1,
            val_auroc,
            per_class_report,
            conf_matrix
        ) = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUROC: {val_auroc:.4f}")
        print(f"Val Precision: {val_prec:.4f} | Val Recall: {val_rec:.4f} | Val F1: {val_f1:.4f}\n")
        
        # 6. Save JSON Log
        history_log.append({
            "epoch": epoch + 1,

            "train_loss": avg_train_loss,
            "train_acc": train_acc,

            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_f1": val_f1,

            "val_auroc": (
                val_auroc
                if not np.isnan(val_auroc)
                else None
            ),

            "per_class_metrics": per_class_report,

            "confusion_matrix": conf_matrix.tolist()
        })
        
        with open(os.path.join(args.output_dir, "log.json"), "w") as f:
            json.dump(history_log, f, indent=4)
            
        # 7. Save Checkpoint for EVERY Epoch
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), # ADD THIS LINE
            'metrics': history_log[-1],
            'config': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'seed': args.seed,
                'num_classes': 14,
                'classes': train_dataset.target_classes
            }
        }
        checkpoint_path = os.path.join(args.output_dir, f"epoch_{epoch+1}.pth")
        torch.save(checkpoint_state, checkpoint_path)
        print(f"-> Saved checkpoint: {checkpoint_path}\n")

if __name__ == "__main__":
    main()