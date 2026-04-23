import torch
import pandas as pd
import os
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torch.utils.data import DataLoader

class ObjDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- AUTOMATIC PATH SEARCHING ---
        # Get filename from CSV (e.g., IMG_7076.png)
        img_name = os.path.basename(row["image_path"])
        # Generate matching label name (e.g., IMG_7076.txt)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        
        # Define possible locations
        train_img_path = os.path.join("data", "train", "images", img_name)
        train_label_path = os.path.join("data", "train", "labels", label_name)
        val_img_path = os.path.join("data", "val", "images", img_name)
        val_label_path = os.path.join("data", "val", "labels", label_name)

        # Check if the file is in Train or Val folder automatically
        if os.path.exists(train_img_path):
            actual_img_path = train_img_path
            actual_label_path = train_label_path
        elif os.path.exists(val_img_path):
            actual_img_path = val_img_path
            actual_label_path = val_label_path
        else:
            raise FileNotFoundError(f"Error: Could not find {img_name} in train or val folders!")

        # Load Image
        img = Image.open(actual_img_path).convert("RGB")
        w, h = img.size
        image = to_tensor(img)

        boxes, labels = [], []
        if os.path.exists(actual_label_path):
            with open(actual_label_path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 5:
                        cls, xc, yc, bw, bh = map(float, parts)
                        
                        # YOLO to Pascal VOC conversion with BOUNDARY CLIPPING
                        # Clipping prevents coordinates from going outside [0, width/height]
                        x1 = max(0, (xc - bw/2) * w)
                        y1 = max(0, (yc - bh/2) * h)
                        x2 = min(w, (xc + bw/2) * w)
                        y2 = min(h, (yc + bh/2) * h)

                        # Only add valid boxes to the list
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(int(cls) + 1) # Class 0 is reserved for background
        
        # Handle images with no objects
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        return image, target

# --- UTILITY FUNCTIONS ---

def collate_fn(batch):
    """Groups images and targets into a tuple for the DataLoader."""
    return tuple(zip(*batch))

def get_dataloaders(args):
    """Splits data and returns Train/Val loaders."""
    csv_file = 'data/CSVs/dataset.csv'
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found at: {csv_file}")
        
    full_df = pd.read_csv(csv_file)
    
    # Randomly split: 80% for training, 20% for validation
    train_df = full_df.sample(frac=0.8, random_state=42)
    val_df = full_df.drop(train_df.index)

    train_dataset = ObjDetectionDataset(train_df)
    val_dataset = ObjDetectionDataset(val_df)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )

    return train_loader, val_loader