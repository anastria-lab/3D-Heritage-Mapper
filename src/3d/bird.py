import cv2
import numpy as np
import glob
import os

# ==========================================
# 1. ΦΟΡΤΩΣΗ CALIBRATION DATA & ΡΥΘΜΙΣΕΙΣ
# ==========================================
# Διαδρομές αρχείων
NPZ_PATH = r'C:/Users/akara/OneDrive/Desktop/D_Chatz/3D-Heritage-Mapper/projects/3d/calibration/camera_calibration_data.npz'
INPUT_FOLDER = r'C:/Users/akara/OneDrive/Desktop/D_Chatz/3D-Heritage-Mapper/projects/3d/data/frames'          # Ο φάκελος με τα 330 αρχικά frames
UNDIST_FOLDER = r'C:/Users/akara/OneDrive/Desktop/D_Chatz/3D-Heritage-Mapper/projects/3d/data/undistorted_frames' # Φάκελος για τα διορθωμένα frames (Για Agisoft)
BEV_FOLDER = r'C:/Users/akara/OneDrive/Desktop/D_Chatz/3D-Heritage-Mapper/projects/3d/data/bev_frames'            # Φάκελος για το Bird's Eye View

# Δημιουργία φακέλων αν δεν υπάρχουν
os.makedirs(UNDIST_FOLDER, exist_ok=True)
os.makedirs(BEV_FOLDER, exist_ok=True)

# Φόρτωση του συμπιεσμένου αρχείου .npz
calib_data = np.load(NPZ_PATH)

# Εξαγωγή των πινάκων K (Intrinsic) και D (Distortion) απευθείας από το .npz
K = calib_data['mtx.npy']
D = calib_data['dist.npy']

print("✓ Το αρχείο Calibration φορτώθηκε επιτυχώς.")

# ==========================================
# 2. INTERACTIVE SELECTION (ΕΠΙΛΟΓΗ 4 ΣΗΜΕΙΩΝ)
# ==========================================
src_pts = []

def click_event(event, x, y, flags, params):
    """Καταγράφει τις συντεταγμένες όταν κάνεις αριστερό κλικ."""
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(src_pts) < 4:
            src_pts.append([x, y])
            print(f"Επιλέχθηκε Σημείο {len(src_pts)}: ({x}, {y})")
            # Σχεδίαση κόκκινου κύκλου στο σημείο click
            cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Select 4 Ground Points (Clockwise)", img_display)

# Ανάγνωση όλων των εικόνων
image_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, '*.jpg')) + 
                     glob.glob(os.path.join(INPUT_FOLDER, '*.png')))

if not image_files:
    raise FileNotFoundError(f"Δεν βρέθηκαν εικόνες στον φάκελο: {INPUT_FOLDER}")

# Παίρνουμε την πρώτη εικόνα για το setup
first_img = cv2.imread(image_files[0])
h, w = first_img.shape[:2]

# Undistort στην πρώτη εικόνα χρησιμοποιώντας τα δεδομένα calibration
first_undist = cv2.undistort(first_img, K, D, None, K)
img_display = first_undist.copy()

print("\n-------------------------------------------------------------")
print("ΟΔΗΓΙΕΣ ΕΠΙΛΟΓΗΣ:")
print("1. Κάντε CLICK σε 4 σημεία στο ΕΔΑΦΟΣ που σχηματίζουν ορθογώνιο.")
print("2. Η σειρά ΠΡΕΠΕΙ να είναι ωρολογιακή:")
print("   - 1ο κλικ: Πάνω-Αριστερά")
print("   - 2ο κλικ: Πάνω-Δεξιά")
print("   - 3ο κλικ: Κάτω-Δεξιά")
print("   - 4ο κλικ: Κάτω-Αριστερά")
print("3. Πατήστε οποιοδήποτε πλήκτρο στο πληκτρολόγιο μόλις τελειώσετε.")
print("-------------------------------------------------------------\n")

# Δημιουργούμε το παράθυρο και του λέμε να είναι "WINDOW_NORMAL" (δηλαδή resizable)
cv2.namedWindow("Select 4 Ground Points (Clockwise)", cv2.WINDOW_NORMAL)

# Το κάνουμε resize σε μια λογική ανάλυση που χωράει σε όλες τις οθόνες (π.χ. 1280x720)
cv2.resizeWindow("Select 4 Ground Points (Clockwise)", 1280, 720)

cv2.setMouseCallback("Select 4 Ground Points (Clockwise)", click_event)
cv2.imshow("Select 4 Ground Points (Clockwise)", img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(src_pts) != 4:
    raise ValueError("Σφάλμα: Πρέπει να επιλέξετε ακριβώς 4 σημεία. Ξανατρέξτε τον κώδικα.")

# Μετατροπή λίστας σημείων σε μορφή numpy array
pts1 = np.float32(src_pts)

# Μεγαλώνουμε λίγο τον καμβά για να χωρέσει η ταμπέλα
# Ορισμός ενός τεράστιου κάθετου καμβά για να χωρέσει το "τέντωμα" του ύψους
bev_w, bev_h = 1200, 2000

# Συμπιέζουμε το παραλληλόγραμμο από τα πλακάκια στο ΚΑΤΩ μέρος του καμβά.
# Έτσι αφήνουμε 1600 pixels κενό χώρο προς τα πάνω για να απλωθεί η ταμπέλα.
pts2 = np.float32([
    [500, 1600],                 # Πάνω-Αριστερά (στο έδαφος)
    [700, 1600],                 # Πάνω-Δεξιά (στο έδαφος)
    [700, 1900],                 # Κάτω-Δεξιά (πιο κοντά στην κάμερα)
    [500, 1900]                  # Κάτω-Αριστερά (πιο κοντά στην κάμερα)
])

# Υπολογισμός Πίνακα Ομογραφίας (H)
H = cv2.getPerspectiveTransform(pts1, pts2)
print("✓ Ο Πίνακας Ομογραφίας (H) για το ακραίο Zoom-Out υπολογίστηκε επιτυχώς.")

# ==========================================
# 3. ΜΑΖΙΚΗ ΕΠΕΞΕΡΓΑΣΙΑ (BATCH PROCESSING)
# ==========================================
print(f"\nΈναρξη επεξεργασίας {len(image_files)} frames...")

for idx, img_path in enumerate(image_files):
    filename = os.path.basename(img_path)
    img = cv2.imread(img_path)
    
    # 1. Undistortion (Αυτές οι εικόνες είναι έτοιμες για το Agisoft)
    undistorted_img = cv2.undistort(img, K, D, None, K)
    cv2.imwrite(os.path.join(UNDIST_FOLDER, filename), undistorted_img)
    
    # 2. Bird's Eye View (Αυτές οι εικόνες είναι το 2D top-down αποτέλεσμα)
    bev_img = cv2.warpPerspective(undistorted_img, H, (bev_w, bev_h))
    cv2.imwrite(os.path.join(BEV_FOLDER, filename), bev_img)
    
    if (idx + 1) % 50 == 0 or (idx + 1) == len(image_files):
        print(f"Επεξεργάστηκαν {idx + 1}/{len(image_files)} εικόνες.")

print("\n🎉 Η διαδικασία ολοκληρώθηκε επιτυχώς!")