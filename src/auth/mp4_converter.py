import os
import subprocess
# Εισάγουμε το config της moviepy για να βρούμε το ενσωματωμένο ffmpeg
import moviepy.config as moviepy_config

def convert_mov_to_mp4(input_path, output_path):
    # Παίρνουμε τη διαδρομή του ffmpeg.exe από το virtual environment σου
    ffmpeg_path = moviepy_config.FFMPEG_BINARY
    
    # Στήνουμε την εντολή FFmpeg για μέγιστη συμβατότητα και κορυφαία ποιότητα
    command = [
        ffmpeg_path, '-y',
        '-i', input_path,
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',        # Μετατρέπει τα χρώματα του iPhone για να παίζουν στα Windows
        '-crf', '18',                 # Κλειδώνει την κορυφαία ποιότητα εικόνας (Lossless αίσθηση)
        '-an',                        # Αφαιρεί τον ήχο για να μην κρασάρει με το Spatial Audio
        output_path
    ]
    
    try:
        # Εκτέλεση της μετατροπής στο παρασκήνιο
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception as e:
        print(f"\n❌ Σφάλμα κατά τη μετατροπή: {e}")
        return False

def main():
    # Ο φάκελος με τα αρχεία .MOV του iPhone 16
    video_folder = r"C:\Users\akara\OneDrive\Desktop\D_Chatz\3D-Heritage-Mapper\projects\3d\data\videos"

    if not os.path.exists(video_folder):
        print(f"Σφάλμα: Ο φάκελος δεν υπάρχει: {video_folder}")
        return

    # Εντοπισμός όλων των .mov αρχείων
    all_files = os.listdir(video_folder)
    mov_files = [f for f in all_files if f.lower().endswith('.mov')]

    if not mov_files:
        print(f"Δεν βρέθηκαν αρχεία .mov στον φάκελο: {video_folder}")
        return

    print(f"--- Βρέθηκαν {len(mov_files)} αρχεία .mov για μετατροπή ---")
    
    # Loop για τη μαζική μετατροπή όλων των βίντεο του φακέλου
    for index, filename in enumerate(mov_files, start=1):
        input_file_path = os.path.join(video_folder, filename)
        
        # Δημιουργία του νέου ονόματος με κατάληξη .mp4
        new_filename = os.path.splitext(filename)[0] + ".mp4"
        output_file_path = os.path.join(video_folder, new_filename)
        
        print(f"🚀 [{index}/{len(mov_files)}] Μετατροπή: {filename} -> {new_filename}...", end="", flush=True)
        
        success = convert_mov_to_mp4(input_file_path, output_file_path)
        if success:
            print(" ✨ Επιτυχία!")
        else:
            print(" ❌ Απέτυχε")

    print("\n--- Όλες οι μετατροπές ολοκληρώθηκαν με κορυφαία ποιότητα! ---")

if __name__ == "__main__":
    main()