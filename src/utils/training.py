from ultralytics import YOLO

def main():
    # Φορτώνουμε το μοντέλο YOLO26 (έκδοση nano για μέγιστη ταχύτητα)
    model = YOLO("yolo26n.pt") 

    print("Ξεκινάει το training του αγάλματος με το YOLO26 στην RTX 4050...")
    
    # Ξεκινάμε την εκπαίδευση
    results = model.train(
        # ΠΡΟΣΟΧΗ: Εδώ βάζουμε το απόλυτο path για το data.yaml
        data=r"C:\Users\akara\OneDrive\Desktop\D_Chatz\3D-Heritage-Mapper\data\mermaid_dataset\LittleMermaid.yolo26\data.yaml", 
        epochs=100,         # 50 γύροι είναι το ιδανικό ξεκίνημα
        imgsz=512,         # Το μέγεθος εικόνας που ορίσαμε και στο Roboflow
        device=0,          # Ενεργοποιεί την NVIDIA κάρτα σου
        workers=2          # Αποτρέπει "κολλήματα" μνήμης στα Windows
    )
    
    print("Η εκπαίδευση ολοκληρώθηκε με επιτυχία!")

if __name__ == '__main__':
    main()