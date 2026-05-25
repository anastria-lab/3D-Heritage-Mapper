# 3D-Heritage-Mapper

An integrated Computer Vision and Geoinformatics pipeline designed to detect urban objects (like traffic lights) under varying lighting and weather conditions, and to reconstruct cultural heritage monuments in 3D using photogrammetry.

## Project Goals
1. **Urban Object Detection:** Test and train machine learning algorithms to identify specific objects across different times of day (morning, noon, night) and weather conditions.
2. **Selective Data Fetching:** Intelligently query massive open autonomous driving datasets (BDD100K, PandaSet, nuScenes) to download *only* the specific frames or scenes needed, saving local storage.
3. **3D Reconstruction:** Extract frames from targeted videos to generate 3D models of monuments using photogrammetry software like Agisoft Metashape.

## Folder Structure
* `data/raw/` - Downloaded video clips and images.
* `data/processed/` - Extracted frames ready for ML or 3D modeling.
* `data/metadata/` - Small JSON/CSV annotation files used to search for specific scenes.
* `src/data_loaders/` - Scripts to securely interact with dataset APIs (Kaggle, nuScenes).
* `src/utils/` - Helper scripts (like video-to-frame extraction).
* `models/` - Saved machine learning weights.

## Technologies Used
* Python (OpenCV, PyTorch, Pandas)
* Kaggle API & nuScenes Devkit
* Agisoft Metashape (External)

## Setup Instructions: Kaggle API
To use the data exploration scripts in this project, you must authenticate with Kaggle:

1. Log in to your [Kaggle](https://www.kaggle.com/) account.
2. Click on your profile picture in the top right and select **Settings**.
3. Scroll down to the **API** section.
4. Locate the **Legacy API Credentials** header and click the **Create New Token** button. *(Note: Using the newer "Generate New Token" button under "API Tokens" will not give you the required JSON file).*
5. This will prompt the download of a `kaggle.json` file.
6. Move this `kaggle.json` file directly into the root directory of this project (`3D-Heritage-Mapper/`).

> **Security Note:** The `.gitignore` file is configured to ignore `kaggle.json`, ensuring your private credentials are never uploaded to GitHub.