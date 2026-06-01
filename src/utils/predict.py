from ultralytics import YOLO
import os

def main():
    # 1. Φορτώνουμε το ΕΚΠΑΙΔΕΥΜΕΝΟ μοντέλο μας (το best.pt)
    model_path = r"C:\Users\akara\OneDrive\Desktop\D_Chatz\3D-Heritage-Mapper\runs\detect\train\weights\best.pt"
    model = YOLO(model_path)

    # 2. Ορίζουμε την εικόνα που θέλουμε να ελέγξουμε
    # ΠΡΟΣΟΧΗ: Βάλε εδώ το path μιας πραγματικής φωτογραφίας που έχεις στο PC σου!
    image_path = r"C:\Users\akara\OneDrive\Desktop\D_Chatz\3D-Heritage-Mapper\data\mermaid_dataset\LittleMermaid.v2i.yolo26\test\images\test_3.jpg"

    if not os.path.exists(image_path):
        print(f"Σφάλμα: Δεν βρέθηκε η εικόνα στη διαδρομή: {image_path}")
        print("Παρακαλώ βάλε μια σωστή διαδρομή εικόνας στο script.")
        return

    print("Το YOLO26 αναλύει την εικόνα...")
    
    # 3. Εκτέλεση της αναγνώρισης
    results = model.predict(
        source=image_path,
        save=True,           # Αποθηκεύει αυτόματα το αποτέλεσμα με το κουτάκι
        imgsz=512,           # Ίδιο μέγεθος με αυτό της εκπαίδευσης
        conf=0.10            # Δείξε μόνο όσα κουτάκια έχουν σιγουριά πάνω από 40%
    )
    
    print("\nΗ αναγνώριση ολοκληρώθηκε!")
    print("Το αποτέλεσμα αποθηκεύτηκε στον φάκελο: runs\\detect\\predict\\")

if __name__ == '__main__':
    main()