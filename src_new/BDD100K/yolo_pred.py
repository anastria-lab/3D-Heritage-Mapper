import os
from ultralytics import YOLO

def run_yolo_on_test_images_smart():
    # Define the subfolders (daytime, fog, night) to run them all together!
    conditions = ["daytime", "fog", "night"]
    base_input = r"projects\bdd\data\test_images"
    output_folder = r"projects\bdd\yolo_predictions_tel"

    model = YOLO("yolo26n.pt") 


    for condition in conditions:
        input_folder = os.path.join(base_input, condition)
        print(f"Starting processing for: {condition.upper()}")

        if not os.path.exists(input_folder):
            print(f"Error: The folder {input_folder} does not exist!")
            continue

        images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(images)} images in {condition}")

        for i, img_name in enumerate(images, 1):
            img_path = os.path.join(input_folder, img_name)
            print(f"[{i}/{len(images)}] processing: {img_name}.")

            results = model.predict(
                source=img_path,
                classes=[9],        # Only Traffic Lights
                conf=0.25,          # Keep predictions with confidence > 25%
                iou=0.45,           # Clean up duplicate boxes
                imgsz=1280,         # Increase resolution
                augment=True,       # Test-Time Augmentation
                save=True,          
                save_txt=True,      
                save_conf=True,    
                project=output_folder, 
                name=condition,    
                exist_ok=True       
            )

if __name__ == "__main__":
    run_yolo_on_test_images_smart()