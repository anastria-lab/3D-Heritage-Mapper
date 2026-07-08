import json
import os
import random
import shutil

# 🔐 Ορισμός του φακέλου για το kaggle.json ΠΡΙΝ γίνει import η βιβλιοθήκη kaggle
os.environ["KAGGLE_CONFIG_DIR"] = os.path.abspath(".")

import kaggle


def download_large_traffic_lights():
    json_path = "projects/bdd/data/metadata/bdd100k_labels_images_val.json"
    output_folder = (
        r"C:\Users\akara\OneDrive\Desktop\D_Chatz\3D-Heritage-Mapper\projects\bdd\data\test_images\night"
    )
    dataset_name = "solesensei/solesensei_bdd100k"

    print(f"🔄 Ανάγνωση μεταδεδομένων από το {json_path}...")

    if not os.path.exists(json_path):
        print(f"❌ Error: Δεν βρέθηκε το αρχείο JSON στο {json_path}")
        return

    with open(json_path, "r") as file:
        data = json.load(file)

    good_images = []

    # 🎯 Όριο μεγέθους: Το φανάρι πρέπει να έχει εμβαδόν τουλάχιστον 1600 pixels (π.χ. 40x40)
    # για να είμαστε σίγουροι ότι είναι κοντινό και καθαρό!
    MIN_BOX_AREA = 1600  

    for image in data:
        weather = image["attributes"].get("weather", "")
        timeofday = image["attributes"].get("timeofday", "")
        image_name = image["name"]

        # Φιλτράρισμα: Μόνο Μέρα και Καθαρός Καιρός
        if timeofday == "daytime" and weather == "clear":
            has_large_light = False
            
            if "labels" in image:
                for label in image["labels"]:
                    if label["category"] == "traffic light":
                        box = label.get("box2d", {})
                        x1, y1 = box.get("x1", 0), box.get("y1", 0)
                        x2, y2 = box.get("x2", 0), box.get("y2", 0)
                        
                        # Υπολογισμός πλάτους, ύψους και εμβαδού
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        
                        # Αν βρούμε έστω και ΕΝΑ μεγάλο φανάρι στην εικόνα, την κρατάμε
                        if area >= MIN_BOX_AREA:
                            has_large_light = True
                            break # Δεν χρειάζεται να ψάξουμε τα υπόλοιπα στην ίδια φωτό
            
            if has_large_light:
                good_images.append(image_name)

    print("\n=============================================")
    print("📊 ΣΤΑΤΙΣΤΙΚΑ ΦΙΛΤΡΑΡΙΣΜΑΤΟΣ (ΜΕΓΑΛΑ ΦΑΝΑΡΙΑ)")
    print("=============================================")
    print(f"☀️  Βρέθηκαν {len(good_images)} πρωινές εικόνες με ΚΟΝΤΙΝΑ/ΜΕΓΑΛΑ φανάρια.")
    print("=============================================\n")

    # Επιλογή 30 τυχαίων από τις "ποιοτικές" εικόνες
    random.seed(42)
    if len(good_images) >= 30:
        target_images = random.sample(good_images, 30)
    else:
        print(f"⚠️  Βρέθηκαν λιγότερες από 30 εικόνες ({len(good_images)}). Θα κατεβούν όλες.")
        target_images = good_images

    # Καθαρισμός του παλιού φακέλου test_images
    if os.path.exists(output_folder):
        print("🧹 Καθαρισμός παλιών εικόνων από τον φάκελο εξόδου...")
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # --- ΑΥΘΕΝΤΙΚΟΠΟΙΗΣΗ KAGGLE ---
    try:
        kaggle.api.authenticate()
        print("✅ Η αυθεντικοποίηση πέτυχε!")
    except Exception as e:
        print(f"❌ Αποτυχία σύνδεσης στο Kaggle API: {e}")
        return

    # --- ΧΑΡΤΟΓΡΑΦΗΣΗ ΔΟΜΗΣ ΚΑΓΚΛΕ ---
    print("\n🔍 Ανίχνευση της δομής φακέλων στο Kaggle...")
    file_mapping = {}
    try:
        dataset_files = kaggle.api.dataset_list_files(dataset_name).files
        for f in dataset_files:
            base_name = os.path.basename(f.name)
            file_mapping[base_name] = f.name
    except Exception as e:
        print(f"⚠️  Χρήση fallback διαδρομών λόγω μεγέθους λίστας.")

    print(f"\n🎯 Έναρξη λήψης των 30 επιλεγμένων εικόνων...")

    success_count = 0
    for i, img_name in enumerate(target_images, 1):
        print(f"📥 [{i}/{len(target_images)}] Λήψη αρχείου: {img_name}...", end="", flush=True)

        if img_name in file_mapping:
            possible_paths = [file_mapping[img_name]]
        else:
            possible_paths = [
                f"bdd100k/bdd100k/images/100k/val/{img_name}",
                f"bdd100k_images_100k/bdd100k/images/100k/val/{img_name}",
                f"bdd100k/images/100k/val/{img_name}",
            ]

        downloaded = False
        for file_path_in_dataset in possible_paths:
            try:
                kaggle.api.dataset_download_file(
                    dataset_name,
                    file_name=file_path_in_dataset,
                    path=output_folder,
                    force=True,
                )

                for root, dirs, files in os.walk(output_folder):
                    if img_name in files and root != output_folder:
                        shutil.move(
                            os.path.join(root, img_name),
                            os.path.join(output_folder, img_name),
                        )

                print(" ✅ Επιτυχία!")
                success_count += 1
                downloaded = True
                break
            except Exception:
                continue

        if not downloaded:
            print(" ❌ Αποτυχία")

    # Τελικός καθαρισμός άχρηστων υποφακέλων
    for item in os.listdir(output_folder):
        item_path = os.path.join(output_folder, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)

    print(f"\n=============================================")
    print(f"✅ Η ΔΙΑΔΙΚΑΣΙΑ ΟΛΟΚΛΗΡΩΘΗΚΕ!")
    print(f"📂 Κατεβάστηκαν επιτυχώς {success_count} πεντακάθαρες πρωινές εικόνες.")
    print(f"📍 Τοποθεσία: {output_folder}")
    print("=============================================\n")


if __name__ == "__main__":
    download_large_traffic_lights()