import json
import os

metadata_path = r"d:\Projects\Political_lens\political_images\metadata.json"

with open(metadata_path, "r") as f:
    images = json.load(f)

# Check if pm_modi_01 is already in it
if not any(img.get("id") == "pm_modi_01" for img in images):
    images.append({
        "id": "pm_modi_01",
        "file_path": r"d:\Projects\Political_lens\political_images\pm_modi_01.png",
        "url": "",
        "label": 0,
        "title": "PM Modi News Incident",
        "subreddit": "unknown"
    })
    
    with open(metadata_path, "w") as f:
        json.dump(images, f, indent=4)
    print("Added pm_modi_01.png to metadata.json")
else:
    print("pm_modi_01.png already exists in metadata.json")
