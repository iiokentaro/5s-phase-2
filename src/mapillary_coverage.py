"""Count the Mapillary images in sample urban and rural areas.

The bbox is a 0.004-degree square centred on each sample point (HALF_BOX).
"""

import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.environ["MAPILLARY_TOKEN"]

# Same sample points used for the OSM pedestrian-infrastructure check.
SAMPLE_POINTS = {
    "thailand_urban_bangkok": (100.55, 13.75),
    "thailand_rural": (101.7, 15.0),
    "maharashtra_urban_pune": (73.85, 18.52),
    "maharashtra_rural": (77.0, 18.3),
}
HALF_BOX = 0.002  # -> 0.004-degree square; the 0.01-degree limit from the
# Data User Guide still 500s in dense areas like central Bangkok, so this is
# tighter than the documented ceiling in practice.


def images_in_bbox(lon: float, lat: float) -> list[dict]:
    bbox = f"{lon - HALF_BOX},{lat - HALF_BOX},{lon + HALF_BOX},{lat + HALF_BOX}"
    resp = requests.get(
        "https://graph.mapillary.com/images",
        params={"access_token": TOKEN, "bbox": bbox, "fields": "id,captured_at", "limit": 50},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def detections_for_image(image_id: str) -> dict:
    resp = requests.get(
        f"https://graph.mapillary.com/{image_id}/detections",
        params={"access_token": TOKEN, "fields": "value"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    first_image_id = None
    for name, (lon, lat) in SAMPLE_POINTS.items():
        images = images_in_bbox(lon, lat)
        years = sorted(
            {
                datetime.fromtimestamp(img["captured_at"] / 1000, tz=timezone.utc).year
                for img in images
                if img.get("captured_at")
            }
        )
        print(f"{name}: {len(images)} images, years={years}")
        if images and first_image_id is None:
            first_image_id = images[0]["id"]

    if first_image_id:
        print(f"\n--- detections sample for image {first_image_id} ---")
        try:
            print(detections_for_image(first_image_id))
        except requests.HTTPError as e:
            print(f"detections request failed: {e}")
