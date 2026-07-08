import os
from ultralytics import YOLO

def run_yolo_on_test_images_smart():
    # 📂 Οι διαδρομές σου
    input_folder = r"C:\Users\akara\OneDrive\Desktop\D_Chatz\3D-Heritage-Mapper\projects\bdd\data\test_images\fog"
    output_folder = r"C:\Users\akara\OneDrive\Desktop\D_Chatz\3D-Heritage-Mapper\projects\bdd\yolo_predictions"
    
    print("🚀 Φόρτωση του YOLO26n...")
    model = YOLO("yolo26n.pt") 

    if not os.path.exists(input_folder):
        print(f"❌ Error: Ο φάκελος {input_folder} δεν υπάρχει!")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"📸 Βρέθηκαν {len(images)} εικόνες για έξυπνο έλεγχο.")
    print("-" * 50)

    for i, img_name in enumerate(images, 1):
        img_path = os.path.join(input_folder, img_name)
        print(f"🧠 [{i}/{len(images)}] Έξυπνη επεξεργασία: {img_name}...")

        # ✨ ΕΔΩ ΓΙΝΕΤΑΙ Η ΜΑΓΕΙΑ ΜΕ ΤΙΣ ΝΕΕΣ ΠΑΡΑΜΕΤΡΟΥΣ ✨
        results = model.predict(
            source=img_path,
            classes=[9],        # Μόνο Traffic Lights
            conf=0.25,          # Κράτα προβλέψεις με σιγουριά > 25%
            iou=0.45,           # Καθαρισμός διπλών κουτιών (Non-Maximum Suppression)
            imgsz=1280,         # 🔥 Αύξηση ανάλυσης για να βλέπει μικρά/μακρινά φανάρια
            augment=True,       # 🔥 Test-Time Augmentation για μεγαλύτερη ακρίβεια
            save=True,          
            project=output_folder, 
            name="run",         
            exist_ok=True       
        )

    print("\n=============================================")
    print("✅ Η ΕΞΥΠΝΗ ΑΝΙΧΝΕΥΣΗ ΟΛΟΚΛΗΡΩΘΗΚΕ!")
    print(f"📂 Δες τα βελτιωμένα bounding boxes εδώ:")
    print(f"📍 {os.path.join(output_folder, 'run')}")
    print("=============================================\n")

if __name__ == "__main__":
    run_yolo_on_test_images_smart()