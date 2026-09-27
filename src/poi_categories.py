"""POI category flags shared by OSM proximity joins and Mapillary map_features."""

from fetch_mapillary_features import OBJECT_VALUES

# Mapillary object_value → category flags (one POI may map to multiple).
MAPILLARY_SCHOOL = {
    "warning--school-zone--g2",
    "regulatory--end-of-school-zone--g1",
}
MAPILLARY_HOSPITAL = {"information--hospital--g1"}
MAPILLARY_BICYCLE = {
    "marking--discrete--symbol--bicycle",
    "object--bike-rack",
    "regulatory--pedestrians-bicycles-permitted--g1",
    "regulatory--shared-path-bicycles-and-pedestrians--g1",
    "regulatory--shared-path-pedestrians-and-bicycles--g1",
    "warning--dual-path-cyclists-and-pedestrians--g1",
}
MAPILLARY_PEDESTRIAN = set(OBJECT_VALUES) - MAPILLARY_BICYCLE - MAPILLARY_SCHOOL - MAPILLARY_HOSPITAL
# Shared-path / dual-path tags are pedestrian as well as bicycle.
MAPILLARY_PEDESTRIAN |= MAPILLARY_BICYCLE & {
    "regulatory--pedestrians-bicycles-permitted--g1",
    "regulatory--shared-path-bicycles-and-pedestrians--g1",
    "regulatory--shared-path-pedestrians-and-bicycles--g1",
    "warning--dual-path-cyclists-and-pedestrians--g1",
}

OSM_BOOL_COLS = ["is_school", "is_hospital", "is_marketplace", "is_shop", "is_bus_stop"]
MAPILLARY_BOOL_COLS = ["map_is_school", "map_is_hospital", "map_is_pedestrian", "map_is_bicycle"]
SEGMENT_BOOL_COLS = ["is_school", "is_hospital", "is_marketplace", "is_shop", "is_bus_stop", "is_pedestrian", "is_bicycle"]

MAPILLARY_JSON_PATHS = {
    "maharashtra": "data/mapillary/map_features_maharashtra.json",
    "thailand": "data/mapillary/map_features_thailand.json",
}


def mapillary_flags(object_value: str) -> dict[str, bool]:
    return {
        "map_is_school": object_value in MAPILLARY_SCHOOL,
        "map_is_hospital": object_value in MAPILLARY_HOSPITAL,
        "map_is_pedestrian": object_value in MAPILLARY_PEDESTRIAN,
        "map_is_bicycle": object_value in MAPILLARY_BICYCLE,
    }
