import os

os.environ["KAGGLE_CONFIG_DIR"] = os.path.abspath(".")

import kaggle


def explore_bdd100k():
    dataset_name = "solesensei/solesensei_bdd100k"
    print(f"Connecting to Kaggle for dataset: {dataset_name}...\n")

    kaggle.api.authenticate()

    # The exact paths to the BDD100K label files inside this Kaggle dataset
    target_files = [
        "bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json",
        "bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json",
    ]

    print(
        "We are bypassing the 150,000 images and targeting the metadata files directly."
    )
    print("-" * 60)
    for file_path in target_files:
        print(f"Targeting: {file_path}")
    print("-" * 60)

    answer = input(
        "\nWould you like to download the smaller 'validation' metadata file (~50MB) to data/metadata/ ? (y/n): "
    )

    if answer.lower() == "y":
        print("\nDownloading validation labels... Please wait.")
        try:
            kaggle.api.dataset_download_file(
                dataset_name,
                file_name=target_files[0],
                path="data/metadata/",
                force=True,
            )
            print("Download complete! Check your data/metadata folder.")

        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("Download cancelled.")


if __name__ == "__main__":
    explore_bdd100k()
