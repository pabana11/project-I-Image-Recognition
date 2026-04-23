import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

# 1. Dataset Class
class MyDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label_path = self.df.iloc[idx]['label_path']
        
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    c, x, y, bw, bh = map(float, line.split())
                    # YOLO to Pascal VOC (Faster R-CNN format)
                    x1 = (x - bw/2) * w
                    y1 = (y - bh/2) * h
                    x2 = (x + bw/2) * w
                    y2 = (y + bh/2) * h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(c) + 1) # +1 for background class

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        
        # Simple transform to Tensor
        from torchvision.transforms import functional as F
        img = F.to_tensor(img)
        
        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == '__main__':
    # Settings
    NUM_EPOCHS = 10
    BATCH_SIZE = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Load CSV and Create Loaders
    csv_path = 'data/csv/dataset.csv'
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} bhetena! Paila create_csv.py run gara.")
        exit()

    full_df = pd.read_csv(csv_path)
    # Split: Train 80%, Val 20%
    train_df = full_df.sample(frac=0.8, random_state=42)
    val_df = full_df.drop(train_df.index)

    train_loader = DataLoader(MyDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(MyDataset(val_df), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 3. Model Setup
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2) # 2 classes: bg + door_handle
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)
    
    train_losses, val_losses = [], []

    print("--- Training Started ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        t_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for imgs, targets in pbar:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            t_loss += losses.item()
            pbar.set_postfix(loss=losses.item())

        avg_t = t_loss / len(train_loader)
        
        # Validation
        v_loss = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = [img.to(device) for img in imgs]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                v_loss += sum(l for l in model(imgs, targets).values()).item()
        
        avg_v = v_loss / len(val_loader)

        

        train_losses.append(avg_t)
        val_losses.append(avg_v)
        print(f"Epoch {epoch+1} Summary: Train Loss={avg_t:.4f}, Val Loss={avg_v:.4f}")

    # 4. Save and Plot
    torch.save(model.state_dict(), "best_model.pth")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS+1), train_losses, label='Training Loss', color='#1f77b4', linewidth=2, marker='o')
    plt.plot(range(1, NUM_EPOCHS+1), val_losses, label='Validation Loss', color='#ff7f0e', linewidth=2, marker='s')
    plt.title('Object Detection Learning Curve (Faster R-CNN)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve.png')
    plt.show()
    print("🚀 Training complete! Graph saved as learning_curve.png")