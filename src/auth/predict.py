from ultralytics import YOLO
import os

def main():

    model_path = r"C:\Users\akara\OneDrive\Desktop\D_Chatz\3D-Heritage-Mapper\projects\auth\models\YOLO26_6k_frames\YOLO26\weights\best.pt"
    model = YOLO(model_path)

    # 2. Ορίζουμε τον ΦΑΚΕΛΟ που περιέχει όλες τις test εικόνες
    folder_path = r"C:\Users\akara\OneDrive\Desktop\D_Chatz\3D-Heritage-Mapper\projects\auth\prediction\images_test"

    output_folder = r"C:\Users\akara\OneDrive\Desktop\D_Chatz\3D-Heritage-Mapper\projects\auth\prediction"

    if not os.path.exists(folder_path):
        print(f"Σφάλμα: Δεν βρέθηκε ο φάκελος στη διαδρομή: {folder_path}")
        return

    print(f"Το YOLO26 ξεκινάει την ομαδική ανάλυση των εικόνων στον φάκελο: {folder_path}...")
    
    # 3. Εκτέλεση της αναγνώρισης για ΟΛΕΣ τις εικόνες μαζί
    results = model.predict(
        source=folder_path,  # Δίνοντας τον φάκελο, διαβάζει τα πάντα μέσα!
        save=True,           # Αποθηκεύει όλα τα αποτελέσματα με τα κουτάκια τους
        imgsz=512,           
        conf=0.25, 
        show_labels=False,    # Κρύβει τα labels (ονόματα και ποσοστά %)
        line_width=2,           # Ανέβασα το conf στο 0.25 μιας και το νέο σου μοντέλο θα είναι πιο σίγουρο
    
        project=output_folder, # Ορίζει τον κεντρικό κατάλογο εξόδου
        name="run_results",   # Ορίζει το όνομα του υποφακέλου μέσα στο project
        exist_ok=True         # Αν ο φάκελος υπάρχει, γράφει από πάνω χωρίς να φτιάχνει run_results2, run_results3...
    )
    print("\nΗ ομαδική αναγνώριση ολοκληρώθηκε!")


if __name__ == '__main__':
    main()