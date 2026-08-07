import os
from ultralytics import YOLO

def run_yolo_on_test_images_smart():
    
    input_folder = r"projects\bdd\data\test_images\fog"
    output_folder = r"projects\bdd\yolo_predictions"

    model = YOLO("yolo26n.pt") 

    if not os.path.exists(input_folder):
        print(f"Error: The folder {input_folder} does not exist!")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(images)} images")

    for i, img_name in enumerate(images, 1):
        img_path = os.path.join(input_folder, img_name)
        print(f"[{i}/{len(images)}] processing: {img_name}.")

        results = model.predict(
            source=img_path,
            classes=[9],        # Only Traffic Lights
            conf=0.25,          # Keep predictions with confidence > 25%
            iou=0.45,           # Clean up duplicate boxes (Non-Maximum Suppression)
            imgsz=1280,         # Increase resolution to see small/distant traffic lights
            augment=True,       #Test-Time Augmentation for higher accuracy
            save=True,          
            project=output_folder, 
            name="run",         
            exist_ok=True       
        )

if __name__ == "__main__":
    run_yolo_on_test_images_smart()