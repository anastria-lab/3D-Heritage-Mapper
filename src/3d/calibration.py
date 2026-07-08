import numpy as np
import cv2
import glob

# 1. ΟΡΙΣΜΟΣ ΔΙΑΣΤΑΣΕΩΝ ΣΚΑΚΙΕΡΑΣ (Εσωτερικές γωνίες από το interface)
CHECKERBOARD = (49, 37)

# Κριτήρια για τη βελτιστοποίηση της ακρίβειας των γωνιών (subpixel accuracy)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Λίστες για την αποθήκευση των 3D και 2D σημείων
objpoints = [] # 3D σημεία στον πραγματικό κόσμο
imgpoints = [] # 2D σημεία στο επίπεδο της εικόνας

# Προετοιμασία των 3D σημείων
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# 2. ΦΟΡΤΩΣΗ ΤΩΝ ΦΩΤΟΓΡΑΦΙΩΝ (Full Path)
images = glob.glob('C:\\Users\\akara\\OneDrive\\Desktop\\D_Chatz\\3D-Heritage-Mapper\\projects\\3d\\calibration\\photos\\*.jpg')

if len(images) == 0:
    print("❌ Σφάλμα: Δεν βρέθηκαν εικόνες! Ελέγξτε αν το path ή η κατάληξη (.jpg) είναι σωστά.")
    exit()

print(f"📸 Βρέθηκαν {len(images)} εικόνες προς επεξεργασία. Έναρξη ανάλυσης στο background...")

gray = None
successful_detections = 0

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"⚠️ Αδυναμία ανάγνωσης του αρχείου: {fname}")
        continue
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Εντοπισμός των γωνιών της σκακιέρας
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                                cv2.CALIB_CB_FAST_CHECK + 
                                                                cv2.CALIB_CB_FILTER_QUADS)
    
    # Αν βρεθούν οι γωνίες, αποθηκεύουμε τα σημεία
    if ret == True:
        successful_detections += 1
        objpoints.append(objp)
        # Βελτιστοποίηση ακρίβειας χωρίς εμφάνιση παραθύρου
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        print(f"✅ [OK] Επιτυχής εντοπισμός σκακιέρας στην εικόνα: {fname}")
    else:
        print(f"❌ [FAIL] Δεν εντοπίστηκε ολόκληρο το pattern (49x37) στην εικόνα: {fname}")

# Έλεγχος αν βρέθηκε έστω και μία σωστή εικόνα για να γίνει το calibration
if successful_detections == 0:
    print("\n❌ Κρίσιμο Σφάλμα: Η σκακιέρα δεν εντοπίστηκε σε ΚΑΜΙΑ εικόνα.")
    print("Σιγουρευτείτε ότι οι γωνίες (49, 37) είναι οι εσωτερικές και ότι η σκακιέρα φαίνεται ολόκληρη και καθαρή.")
    exit()

# 3. ΕΚΤΕΛΕΣΗ CALIBRATION
print(f"\n⏳ Υπολογισμός παραμέτρων κάμερας με βάση {successful_detections} έγκυρες εικόνες... Παρακαλώ περιμένετε.")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 4. ΕΜΦΑΝΙΣΗ ΚΑΙ ΑΠΟΘΗΚΕΥΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ
print("\n=== ΑΠΟΤΕΛΕΣΜΑΤΑ CALIBRATION ===")
print("\n1. Πίνακας Κάμερας (Intrinsic Matrix - mtx):")
print(mtx)
print("\n2. Συντελεστές Παραμόρφωσης (Distortion Coefficients - dist):")
print(dist)

# Ορισμός full path για το output
output_folder = 'C:\\Users\\akara\\OneDrive\\Desktop\\D_Chatz\\3D-Heritage-Mapper\\projects\\3d\\calibration\\'

# Αποθήκευση σε binary μορφή .npz (για Python/YOLO)
np.savez(output_folder + 'camera_calibration_data.npz', mtx=mtx, dist=dist)
print(f"\n💾 Τα binary δεδομένα αποθηκεύτηκαν στο: {output_folder}camera_calibration_data.npz")

# Αποθήκευση σε text αρχείο (.txt) για εύκολη ανάγνωση
with open(output_folder + 'calibration_results.txt', 'w', encoding='utf-8') as f:
    f.write("=== ΑΠΟΤΕΛΕΣΜΑΤΑ CALIBRATION ΚΑΜΕΡΑΣ ===\n\n")
    f.write("1. Πίνακας Κάμερας (Intrinsic Matrix - mtx):\n")
    f.write(np.array2string(mtx) + "\n\n")
    f.write("2. Συντελεστές Παραμόρφωσης (Distortion Coefficients - dist):\n")
    f.write(np.array2string(dist) + "\n")

print(f"📝 Τα αποτελέσματα σε μορφή κειμένου αποθηκεύτηκαν στο: {output_folder}calibration_results.txt")