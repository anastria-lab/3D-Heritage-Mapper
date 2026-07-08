import cv2
import os

def extract_frames_by_seconds(video_path, output_folder, start_sec, end_sec, frames_per_sec=2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Άνοιγμα του βίντεο
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Σφάλμα: Δεν είναι δυνατό το άνοιγμα του βίντεο {video_path}")
        return

    # Χαρακτηριστικά βίντεο
    fps = cap.get(cv2.CAP_PROP_FPS)          
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = total_frames / fps

    print(f"🎬 Πληροφορίες Βίντεο: FPS={fps:.2f} | Συνολική Διάρκεια={duration_sec:.2f}s")
    print(f"⏳ Εξαγωγή από {start_sec}s έως {end_sec}s (Σύνολο: {end_sec - start_sec} δευτερόλεπτα)")

    # Έλεγχοι ασφαλείας για τα όρια
    if start_sec > duration_sec:
        print("❌ Σφάλμα: Ο χρόνος έναρξης ξεπερνάει τη διάρκεια του βίντεο!")
        cap.release()
        return

    if end_sec > duration_sec:
        print(f"⚠️ Προειδοποίηση: Ο χρόνος λήξης ξεπερνά το βίντεο. Θα σταματήσει στο τέλος ({duration_sec:.2f}s).")
        end_sec = duration_sec

    # Κάθε πόσα frames του βίντεο κρατάμε 1 frame (π.χ. αν FPS=30 και θέλουμε 2 frames/sec -> interval=15)
    frame_interval = int(fps / frames_per_sec)
    if frame_interval < 1:
        frame_interval = 1

    # Μετακίνηση στο frame έναρξης
    start_frame_idx = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

    current_frame_idx = start_frame_idx
    end_frame_idx = int(end_sec * fps)
    saved_count = 0

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    while current_frame_idx <= end_frame_idx:
        ret, frame = cap.read()
        if not ret:
            break  # Τέλος βίντεο

        # Κρατάμε μόνο τα frames που συμπίπτουν με το interval
        if (current_frame_idx - start_frame_idx) % frame_interval == 0:
            current_time_sec = current_frame_idx / fps
            
            # Ονοματοδοσία με το ακριβές δευτερόλεπτο (π.χ. IMG_0031_sec_24.50_1.jpg)
            frame_filename = f"{video_name}_sec_{current_time_sec:.2f}_{saved_count+1}.jpg"
            frame_output_path = os.path.join(output_folder, frame_filename)
            
            cv2.imwrite(frame_output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved_count += 1

        current_frame_idx += 1

    cap.release()
    print(f"✨ Ολοκληρώθηκε! Εξήχθησαν {saved_count} frames στο φάκελο: {output_folder}")

def main():
    # --- ΡΥΘΜΙΣΕΙΣ ---
    # 1. Βίντεο εισόδου
    video_input = r"C:\Users\akara\OneDrive\Desktop\D_Chatz\3D-Heritage-Mapper\projects\3d\data\videos\IMG_0054.mp4"
    
    # 2. Φάκελος εξαγωγής
    output_directory = r"C:\Users\akara\OneDrive\Desktop\D_Chatz\3D-Heritage-Mapper\projects\3d\data\frames\walking"
    
    # 3. ΧΡΟΝΙΚΟ ΔΙΑΣΤΗΜΑ ΣΕ ΔΕΥΤΕΡΟΛΕΠΤΑ
    start_second = 0    # Από 0:24
    end_second = 150   # Έως 0:49
    
    # 4. Πόσα frames ανά δευτερόλεπτο
    fps_to_extract = 2  

    extract_frames_by_seconds(video_input, output_directory, start_second, end_second, fps_to_extract)

if __name__ == "__main__":
    main()