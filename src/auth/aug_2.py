import os
import cv2
import albumentations as A
import numpy as np
from tqdm import tqdm

# ---- ΟΡΙΣΕ ΤΟΥΣ ΦΑΚΕΛΟΥΣ ΣΟΥ ----
images_dir = "C:/Users/akara/OneDrive/Desktop/D_Chatz/3D-Heritage-Mapper/projects/auth/data/training_dataset/CH_test_1/train/images"  # Ο φάκελος με τις τωρινές 1600 εικόνες
labels_dir = "C:/Users/akara/OneDrive/Desktop/D_Chatz/3D-Heritage-Mapper/projects/auth/data/training_dataset/CH_test_1/train/labels"  # Ο φάκελος με τα τωρινά 1600 .txt αρχεία

output_images_dir = "C:/Users/akara/OneDrive/Desktop/D_Chatz/3D-Heritage-Mapper/projects/auth/data/training_dataset/CH_test_2/train/images"
output_labels_dir = "C:/Users/akara/OneDrive/Desktop/D_Chatz/3D-Heritage-Mapper/projects/auth/data/training_dataset/CH_test_2/train/labels"


os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

def read_labels(label_path):
    if not os.path.exists(label_path): return []
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def save_labels(output_path, lines):
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
print(f"Βρέθηκαν {len(image_files)} εικόνες segmentation. Ξεκινάμε τη σωστή μετατροπή...")

for img_file in tqdm(image_files):
    base_name = os.path.splitext(img_file)[0]
    img_path = os.path.join(images_dir, img_file)
    label_path = os.path.join(labels_dir, base_name + ".txt")
    
    img = cv2.imread(img_path)
    if img is None: continue
    labels = read_labels(label_path)
    
    if not labels: continue
    
    # ---- 1. Αρχική Εικόνα & Brightness (Αντιγραφή ως έχει, χωρίς σπάσιμο) ----
    cv2.imwrite(os.path.join(output_images_dir, img_file), img)
    save_labels(os.path.join(output_labels_dir, base_name + ".txt"), labels)
    
    img_bright = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
    cv2.imwrite(os.path.join(output_images_dir, base_name + "_bright.jpg"), img_bright)
    save_labels(os.path.join(output_labels_dir, base_name + "_bright.txt"), labels)
    
    # ---- 2. Υπολογισμός Horizontal Flip για Polygons (Όλα τα X αλλάζουν) ----
    flipped_labels = []
    for line in labels:
        parts = line.split()
        class_id = parts[0]
        coords = parts[1:]
        
        new_coords = []
        # Τα δεδομένα είναι ζευγάρια: X = coords[i], Y = coords[i+1]
        for i in range(0, len(coords), 2):
            x = float(coords[i])
            y = float(coords[i+1])
            
            new_x = 1.0 - x  # Καθρεφτισμός του X
            new_coords.append(f"{new_x:.6f}")
            new_coords.append(f"{y:.6f}")
            
        flipped_labels.append(f"{class_id} " + " ".join(new_coords))
        
    # ---- 3. Αποθήκευση των Flipped Αρχείων ----
    img_flip = cv2.flip(img, 1)
    cv2.imwrite(os.path.join(output_images_dir, base_name + "_flip.jpg"), img_flip)
    save_labels(os.path.join(output_labels_dir, base_name + "_flip.txt"), flipped_labels)
    
    img_flip_bright = cv2.convertScaleAbs(img_flip, alpha=1.2, beta=30)
    cv2.imwrite(os.path.join(output_images_dir, base_name + "_flip_bright.jpg"), img_flip_bright)
    save_labels(os.path.join(output_labels_dir, base_name + "_flip_bright.txt"), flipped_labels)

print("\nΤΕΛΟΣ! Όλα τα πολύγωνα μετατράπηκαν σωστά χωρίς να χαθεί καμία συντεταγμένη!")