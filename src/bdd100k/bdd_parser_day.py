import json
import os
import random
import shutil

#Define the folder for kaggle.json
os.environ["KAGGLE_CONFIG_DIR"] = os.path.abspath(".")

import kaggle


def download_large_traffic_lights():
    json_path = "projects/bdd/data/metadata/bdd100k_labels_images_val.json"
    output_folder = (
        r"projects\bdd\data\test_images\daytime"
    )
    dataset_name = "solesensei/solesensei_bdd100k"

    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    with open(json_path, "r") as file:
        data = json.load(file)

    good_images = []

    MIN_BOX_AREA = 1600  

    for image in data:
        weather = image["attributes"].get("weather", "")
        timeofday = image["attributes"].get("timeofday", "")
        image_name = image["name"]

        #Daytime and clear weather
        if timeofday == "daytime" and weather == "clear":
            has_large_light = False
            
            if "labels" in image:
                for label in image["labels"]:
                    if label["category"] == "traffic light":
                        box = label.get("box2d", {})
                        x1, y1 = box.get("x1", 0), box.get("y1", 0)
                        x2, y2 = box.get("x2", 0), box.get("y2", 0)
                        
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        
                        # If we find even ONE large traffic light in the image, we keep it
                        if area >= MIN_BOX_AREA:
                            has_large_light = True
                            break 
            
            if has_large_light:
                good_images.append(image_name)

    print(f"Found {len(good_images)} daytime images with traffic lights.")

    # Select 30 random from the images
    random.seed(42)
    if len(good_images) >= 30:
        target_images = random.sample(good_images, 30)
    else:
        print(f"⚠️  Found fewer than 30 images ({len(good_images)}). All will be downloaded.")
        target_images = good_images

    # Clean up the old test_images folder
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    #KAGGLE AUTHENTICATION
    try:
        kaggle.api.authenticate()
        print("✅ Authentication successful!")
    except Exception as e:
        print(f"❌ Failed to connect to Kaggle API: {e}")
        return

    #KAGGLE DIRECTORY MAPPING
    print("\n Detecting folder structure on Kaggle...")
    file_mapping = {}
    try:
        dataset_files = kaggle.api.dataset_list_files(dataset_name).files
        for f in dataset_files:
            base_name = os.path.basename(f.name)
            file_mapping[base_name] = f.name
    except Exception as e:
        print(f"⚠️  Using fallback paths due to list size.")

    print(f"\n Starting download")

    success_count = 0
    for i, img_name in enumerate(target_images, 1):
        print(f"📥 [{i}/{len(target_images)}] Downloading file: {img_name}...", end="", flush=True)

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

                print("Success!")
                success_count += 1
                downloaded = True
                break
            except Exception:
                continue

        if not downloaded:
            print(" ❌ Failure")

    # Final cleanup of useless subfolders
    for item in os.listdir(output_folder):
        item_path = os.path.join(output_folder, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)

    print(f" Successfully downloaded {success_count} daytime images.")
    print(f" Location: {output_folder}")

if __name__ == "__main__":
    download_large_traffic_lights()