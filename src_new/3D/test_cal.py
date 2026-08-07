import cv2
import numpy as np

# 1. Φόρτωση των δεδομένων που μόλις βγάλαμε
data = np.load('C:\\Users\\akara\\OneDrive\\Desktop\\D_Chatz\\3D-Heritage-Mapper\\projects\\3d\\calibration\\camera_calibration_data.npz')
mtx = data['mtx']
dist = data['dist']

# 2. Διάβασμα ΜΙΑΣ φωτογραφίας σου (βάλε το όνομα μιας υπαρκτής εικόνας)
img_path = 'C:\\Users\\akara\\OneDrive\\Desktop\\D_Chatz\\3D-Heritage-Mapper\\projects\\3d\\calibration\\photos\\IMG_2294.jpg'
img = cv2.imread(img_path)

if img is None:
    print("Δεν βρέθηκε η εικόνα δοκιμής!")
    exit()

# 3. Εφαρμογή του Undistort
undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

# 4. Αποθήκευση της διορθωμένης εικόνας
output_path = 'C:\\Users\\akara\\OneDrive\\Desktop\\D_Chatz\\3D-Heritage-Mapper\\projects\\3d\\calibration\\corrected_image.jpg'
cv2.imwrite(output_path, undistorted_img)

print(f"Η διορθωμένη εικόνα αποθηκεύτηκε στο: {output_path}")
print("Άνοιξε τις δύο εικόνες δίπλα-δίπλα. Οι γραμμές της σκακιέρας στη διορθωμένη πρέπει να είναι απόλυτα ίσιες!")