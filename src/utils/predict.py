from ultralytics import YOLO
import os

def main():

    model_path = r"C:\Users\akara\OneDrive\Desktop\D_Chatz\3D-Heritage-Mapper\runs\detect\raw_uncompressed_100_e\weights\best.pt"
    model = YOLO(model_path)

    # 2. Ορίζουμε τον ΦΑΚΕΛΟ που περιέχει όλες τις test εικόνες
    folder_path = r"C:\Users\akara\OneDrive\Desktop\D_Chatz\3D-Heritage-Mapper\data\mermaid_dataset\LittleMermaid.yolo26\test\images"

    if not os.path.exists(folder_path):
        print(f"Σφάλμα: Δεν βρέθηκε ο φάκελος στη διαδρομή: {folder_path}")
        return

    print(f"Το YOLO26 ξεκινάει την ομαδική ανάλυση των εικόνων στον φάκελο: {folder_path}...")
    
    # 3. Εκτέλεση της αναγνώρισης για ΟΛΕΣ τις εικόνες μαζί
    results = model.predict(
        source=folder_path,  # Δίνοντας τον φάκελο, διαβάζει τα πάντα μέσα!
        save=True,           # Αποθηκεύει όλα τα αποτελέσματα με τα κουτάκια τους
        imgsz=512,           
        conf=0.25            # Ανέβασα το conf στο 0.25 μιας και το νέο σου μοντέλο θα είναι πιο σίγουρο
    )
    
    print("\nΗ ομαδική αναγνώριση ολοκληρώθηκε!")
    print("Όλες οι επεξεργασμένες εικόνες αποθηκεύτηκαν στον φάκελο: runs\\detect\\predict\\")

if __name__ == '__main__':
    main()