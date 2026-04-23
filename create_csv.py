import os
import pandas as pd

# --- Configuration ---
train_img_dir = 'data/train/images'
train_lbl_dir = 'data/train/labels'
val_img_dir = 'data/val/images'
val_lbl_dir = 'data/val/labels'

output_csv = 'data/csv/dataset.csv' # for saving the data set

data = []

def scan_folder(img_dir, lbl_dir):
    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        print(f"⚠️ Warning: Missing {img_dir} or {lbl_dir}")
        return
    
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_map = {os.path.splitext(f)[0].strip(): f for f in image_files}
    
    count = 0
    for lbl_name in os.listdir(lbl_dir):
        if lbl_name.endswith('.txt'):
            base_name = os.path.splitext(lbl_name)[0].strip()
            if base_name in image_map:
                data.append({
                    'image_path': os.path.join(img_dir, image_map[base_name]),
                    'label_path': os.path.join(lbl_dir, lbl_name)
                })
                count += 1
    print(f"✅ Found {count} pairs in {img_dir}")

# scan both train and val folders
print("--- Scanning Folders ---")
scan_folder(train_img_dir, train_lbl_dir)
scan_folder(val_img_dir, val_lbl_dir)

# Save to CSV
df = pd.DataFrame(data)
if not df.empty:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n🚀 Total {len(df)} images linked successfully!")
    print(f"📁 CSV saved at: {output_csv}")
else:
    print("\n❌ Error: 0 matches found! Check if file names match exactly.")