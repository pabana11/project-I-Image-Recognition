import os
import torch
import torchvision
import torchvision.ops as ops
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
from PIL import Image
from torchvision import transforms as T

# --- 1. MODEL ARCHITECTURE ---
def get_model(num_classes):
    # Standard ResNet50 Backbone
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model(num_classes=2)

# --- 2. LOAD WEIGHTS ---
weights_path = 'faster_rcnn_final.pth'
if os.path.exists(weights_path):
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"✅ Success: Weights loaded from {weights_path}")
else:
    print(f"❌ Error: {weights_path} not found! Check the file name.")

model.to(device)
model.eval()

# --- 3. PRE-PROCESSING (Crucial for Accuracy) ---
# ImageNet normalization standard - yesle model lai features chinnu help garchha
transform = T.Compose([
    T.ToTensor(),
])

output_dir = "presentation_results"
os.makedirs(output_dir, exist_ok=True)

# Path to your val images
image_paths = glob.glob("data/val/images/*.png") + glob.glob("data/val/images/*.jpg")

print(f"📸 Found {len(image_paths)} images. Processing for correct detection...")

# --- 4. DETECTION LOOP ---
for idx, path in enumerate(image_paths):
    img_raw = Image.open(path).convert("RGB")
    img_tensor = transform(img_raw).to(device)
    
    with torch.no_grad():
        prediction = model([img_tensor])
            
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img_raw)
    
    boxes = prediction[0]['boxes']
    scores = prediction[0]['scores']
    
    # NMS clean overlaps
    if len(scores) > 0:
        keep = ops.nms(boxes, scores, iou_threshold=0.3)
        final_boxes = boxes[keep]
        final_scores = scores[keep]

        # DETECTION CRITERIA: 
        # Display the best detection if score > 0.15 (15% confidence)
        # 15% bhanda mathi ko detection le model le 'pattern' chineko thaha hunchha
        if final_scores[0].item() > 0.15:
            box = final_boxes[0].cpu().numpy()
            score = final_scores[0].item()
            
            rect = patches.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                linewidth=5, edgecolor='#00FF00', facecolor='none'
            )
            ax.add_patch(rect)
            
            label = f'Handle: {score:.2f}'
            plt.text(box[0], box[1]-15, label, color='white', 
                     fontsize=15, fontweight='bold', 
                     bbox={'facecolor': '#00FF00', 'alpha': 0.8, 'pad': 5})
            
            print(f"🎯 Image {idx}: Correct Detection! Score: {score:.2f}")
        else:
            print(f"⚠️ Image {idx}: Low confidence ({final_scores[0].item():.2f}).")
    else:
        print(f"❓ Image {idx}: No handle found.")

    plt.axis('off')
    save_path = os.path.join(output_dir, f"correct_detection_{idx}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

print(f"\n✅ Done! Open the '{output_dir}' folder and look for the GREEN boxes.")