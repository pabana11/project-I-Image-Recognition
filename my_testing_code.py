import torch
import torchvision
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt
import torchvision.transforms as T

# ==========================================
# 1. PATHS & DIRECTORY SETUP
# ==========================================
base_path = os.getcwd() 
model_path = os.path.join(base_path, 'best_model.pth')
image_folder = os.path.join(base_path, 'data', 'images')

# Define and create the output folder for saved images
output_folder = os.path.join(base_path, 'detected_results')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created directory: {output_folder}")

# ==========================================
# 2. MODEL LOADING
# ==========================================
# Initialize model architecture (Faster R-CNN with ResNet-50)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
# Load the trained weights
model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
model.eval() # Set model to evaluation mode
transform = T.Compose([T.ToTensor()])

# ==========================================
# 3. DETECTION, DRAWING, AND SAVING
# ==========================================
# Get all jpeg images from the source folder
image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.jpeg')])

for img_name in image_files:
    img_path = os.path.join(image_folder, img_name)
    img = Image.open(img_path).convert("RGB")
    width, height = img.size
    img_tensor = transform(img).unsqueeze(0)
    
    # Perform detection
    with torch.no_grad():
        prediction = model(img_tensor)
    
    draw = ImageDraw.Draw(img)
    boxes = prediction[0]['boxes']
    
    if len(boxes) > 0:
        # Take the top detection box
        box = boxes[0].numpy()
        
        # Calculate the center of the predicted box
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        
        # Define a custom box size (50% of image dimensions as requested)
        box_width = width * 0.50 
        box_height = height * 0.50
        
        # Create coordinates for the new centered box
        new_box = [
            center_x - (box_width / 2),
            center_y - (box_height / 2),
            center_x + (box_width / 2),
            center_y + (box_height / 2)
        ]
        
        # Draw the final red box on the image
        draw.rectangle(new_box, outline="red", width=20)
    
    # --- SAVE THE IMAGE ---
    save_name = f"detected_{img_name}"
    save_path = os.path.join(output_folder, save_name)
    img.save(save_path) 
    print(f"✅ Success: Saved to {save_path}")
    
    # Display the result in the console/plot window
    plt.figure(figsize=(10,7))
    plt.imshow(img)
    plt.title(f"Detected Handle: {img_name}")
    plt.axis('off')
    plt.show()

print(f"\nAll images have been saved in: {output_folder}")