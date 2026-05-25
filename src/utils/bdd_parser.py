import json
import os


def analyze_bdd_labels():
    json_path = "data/metadata/bdd100k_labels_images_val.json"

    print(f"Loading BDD100K metadata from {json_path}...")

    if not os.path.exists(json_path):
        print("Error: Could not find the JSON file. Make sure it's in data/metadata/")
        return

    with open(json_path, "r") as file:
        data = json.load(file)

    print(f"Successfully loaded {len(data)} images!\n")

    fog_traffic_lights = 0
    night_traffic_lights = 0

    print("Searching for Traffic Lights in Fog and Night conditions...")

    for image in data:
        weather = image["attributes"].get("weather", "")
        timeofday = image["attributes"].get("timeofday", "")

        has_traffic_light = False
        if "labels" in image:
            for label in image["labels"]:
                if label["category"] == "traffic light":
                    has_traffic_light = True
                    break  # We found one, stop checking labels for this image

        # If it has a traffic light, check the weather/time
        if has_traffic_light:
            if weather == "foggy":
                fog_traffic_lights += 1
                # Let's print the first one we find as an example!
                if fog_traffic_lights == 1:
                    print(f"-> Found a Foggy Traffic Light in image: {image['name']}")

            if timeofday == "night":
                night_traffic_lights += 1

    print("\n--- SEARCH RESULTS ---")
    print(f"Images with Traffic Lights in FOG: {fog_traffic_lights}")
    print(f"Images with Traffic Lights at NIGHT: {night_traffic_lights}")


if __name__ == "__main__":
    analyze_bdd_labels()
