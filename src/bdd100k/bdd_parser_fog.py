import json
import os
import random
import shutil

# 🔐 Ορισμός του φακέλου για το kaggle.json ΠΡΙΝ γίνει import η βιβλιοθήκη kaggle
os.environ["KAGGLE_CONFIG_DIR"] = os.path.abspath(".")

import kaggle


def download_foggy_traffic_lights():
    json_path = "projects/bdd/data/metadata/bdd100k_labels_images_val.json"
    output_folder = (
        r"C:\Users\akara\OneDrive\Desktop\D_Chatz\3D-Heritage-Mapper\projects\bdd\data\test_images\fog"
    )
    dataset_name = "solesensei/solesensei_bdd100k"

    print(f"🔄 Ανάγνωση μεταδεδομένων από το {json_path}...")

    if not os.path.exists(json_path):
        print(f"❌ Error: Δεν βρέθηκε το αρχείο JSON στο {json_path}")
        return

    with open(json_path, "r") as file:
        data = json.load(file)

    good_images = []

    for image in data:
        weather = image["attributes"].get("weather", "")
        image_name = image["name"]

        # 🎯 Φιλτράρισμα: Μόνο εικόνες με ομίχλη (Foggy)
        if weather == "foggy":
            has_traffic_light = False
            
            if "labels" in image:
                for label in image["labels"]:
                    # Μας αρκεί να υπάρχει έστω και ένα φανάρι, ανεξαρτήτως μεγέθους
                    if label["category"] == "traffic light":
                        has_traffic_light = True
                        break 
            
            if has_traffic_light:
                good_images.append(image_name)

    print("\n=============================================")
    print("📊 ΣΤΑΤΙΣΤΙΚΑ ΦΙΛΤΡΑΡΙΣΜΑΤΟΣ (ΦΑΝΑΡΙΑ ΣΕ ΟΜΙΧΛΗ)")
    print("=============================================")
    print(f"☁️  Βρέθηκαν {len(good_images)} συνολικές εικόνες με ομίχλη που περιέχουν φανάρια.")
    print("=============================================\n")

    # Επιλογή 30 τυχαίων εικόνων
    random.seed(42)
    if len(good_images) >= 30:
        target_images = random.sample(good_images, 30)
    else:
        print(f"⚠️  Βρέθηκαν λιγότερες από 30 εικόνες ({len(good_images)}). Θα κατεβούν όλες.")
        target_images = good_images

    if os.path.exists(output_folder):
        print("🧹 Καθαρισμός παλιών αρχείων από τον φάκελο fog...")
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

    print(f"\n🎯 Έναρξη λήψης των {len(target_images)} επιλεγμένων εικόνων...")

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

    # Τελικός καθαρισμός
    for item in os.listdir(output_folder):
        item_path = os.path.join(output_folder, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)

    print(f"\n=============================================")
    print(f"✅ Η ΔΙΑΔΙΚΑΣΙΑ ΟΛΟΚΛΗΡΩΘΗΚΕ!")
    print(f"📂 Κατεβάστηκαν επιτυχώς {success_count} εικόνες με ομίχλη.")
    print(f"📍 Τοποθεσία: {output_folder}")
    print("=============================================\n")


if __name__ == "__main__":
    download_foggy_traffic_lights()