import torch
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

def get_model(num_classes=2):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# 1. Load Original Image
img_path = "data/val/images/IMG_7190.png" 
orig_image = cv2.imread(img_path)
if orig_image is None:
    raise FileNotFoundError("Image path milena, check gara!")

orig_h, orig_w = orig_image.shape[:2]

# 2. Resize for Detection (Speed up and Memory Save)

input_size = 640 
image_resized = cv2.resize(orig_image, (input_size, input_size))
image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
img_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(device)

# 3. Predict
with torch.no_grad():
    prediction = model(img_tensor)

# 4. Scaling Logic (Crucial Fix)

scale_x = orig_w / input_size
scale_y = orig_h / input_size

boxes = prediction[0]['boxes']
scores = prediction[0]['scores']

score_thresh = 0.5 
image_copy = orig_image.copy() 

for i in range(len(boxes)):
    score = scores[i].item()
    if score > score_thresh:
        box = boxes[i].cpu().numpy()
        
        
        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)

        
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(image_copy, f"Handle: {score:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        print(f"✅ Handle Detected! Score: {score:.2f}")

# 5. Final Display

display_img = cv2.resize(image_copy, (800, 600)) 
cv2.imshow("Detection Result", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()