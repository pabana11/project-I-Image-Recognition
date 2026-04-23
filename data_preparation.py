import os
import pandas as pd

# Paths
image_dir = 'Data/images'
label_dir = 'Data/labels/train'
csv_folder = 'Data/CSVs'
output_csv = 'Data/CSVs/dataset.csv'


if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

# List all image files
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

rows = []

# Loop over images
for img in image_files:
   
    filename_no_ext = os.path.splitext(img)[0]
    label_file = filename_no_ext + '.txt'

    img_path = os.path.join(image_dir, img)
    label_path = os.path.join(label_dir, label_file)

    if os.path.exists(label_path):
        rows.append([img_path, label_path])

# Create DataFrame and save CSV
df = pd.DataFrame(rows, columns=['images', 'labels'])
df.to_csv(output_csv, index=False)

print(f"Matched pairs: {len(rows)}")
print(df.head())
