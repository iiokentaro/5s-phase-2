# QUICKSTART

This repository holds the code, the documentation and the data files needed to
reproduce the deliverables. Mapillary and TomTom data may not be redistributed, so
those files stay out of it. This page says what runs as it stands, and where to put
the files that are missing.

Every command below runs from the repository root. On the machine this was prepared
on that is `/Users/kentaroiio/Documents/github/5s-phase-2`; substitute the path your
clone landed in.

## Setup

```bash
cd /Users/kentaroiio/Documents/github/5s-phase-2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## What runs as it stands

One command rebuilds every deliverable, in about a minute:

```bash
cd /Users/kentaroiio/Documents/github/5s-phase-2
python src/quick_reproduce.py
```

It writes to `outputs/`: `segments_priority.parquet`, `segments_priority.gpkg`,
`segments_priority.gdb.zip`, the four priority CSV lists
(`priority_review_needed.csv`, `priority_field_check.csv`, `priority_urban.csv`,
`priority_rural.csv`), `priority_map.html`, `priority_map_static.png`, and it prints
the sensitivity tables.

The same result comes from the browser panel:

```bash
cd /Users/kentaroiio/Documents/github/5s-phase-2
python src/serve_map.py --enable-pipeline
```

That opens http://localhost:8000/ with the map on the left and a run panel on the
right. The **Quick** preset is the default; press **Run**. Steps whose input files
are absent report the missing path and are skipped, so the run reaches the end.
Without `--enable-pipeline` the same command serves the map alone.

These are the files the Quick path reads, and all of them are in this repository:

| File | Used by |
|---|---|
| `data/processed/segments_v_safe.parquet` | every deliverable; holds V_safe, exposure, misalignment, the Speed Safety Score, the priority class and the review track |
| `docs/AADT Scaling.xlsx` | the AADT estimate (`src/aadt_estimation.py`) |
| `data/processed/pois_{country}.parquet`, `overture_pois_{country}.parquet`, `osm_pois_{country}.parquet` | the school markers on `outputs/priority_map.html` (`src/poi_sources.py`) |
| `data/processed/osm_junctions_{country}.parquet` | the junction markers on the same map (`src/junction_speed_cap.py`) |

`build_tiles` is the one Quick step that needs something the repository cannot carry:
tippecanoe, which rebuilds `docs/segments_priority.pmtiles`. Install it with
`brew install tippecanoe` (macOS) if you want to rebuild the tiles; the committed
tiles serve the map without it.

`data/raw/ADB_Innovation_Maharashtra.geojson` and
`data/raw/ADB_Innovation_Thailand.geojson` are here too. They are the ADB source
network, and they are the input to the full rebuild.

## Files this repository does not carry

The directory layout under `data/` is in place, with empty directories kept by
`.gitkeep`. Put each file at the path in the first column; the scripts create any
directory that is absent.

| Path | What it is | How to obtain it |
|---|---|---|
| `.env` | `MAPILLARY_TOKEN=...`, read by the Mapillary scripts | your own Mapillary API token |
| `data/mapillary/map_features_{country}.json` | Mapillary map features, 26 VRU classes | `python src/fetch_mapillary_features.py` |
| `data/mapillary/tile_cover/`, `vector_tiles/`, `filtered_map_features/` | decoded Mapillary z14 tiles along the corridor | `python src/extract_map_features.py` |
| `data/mapillary/filtered_image_ids/{region}/` | Mapillary image IDs within 10 m of the roads | `python src/fetch_image_ids.py` |
| `data/mapillary/central_image_ids/{region}_central_image_ids.csv` | the image nearest each segment midpoint | `python src/central_image_ids.py` |
| `data/mapillary/detected_objects/{region}/`, `detected_object_points/{Region}/` | object classes Mapillary detected around each central image | `python src/fetch_detected_objects.py` |
| `data/mapillary/mapillary_detectable_objects.csv` | the object catalogue with the per-class weights | shared on request; `src/segment_detected_objects.py` reads it |
| `data/mapillary/segment_detected_objects/` | the weights summed onto segments | `python src/segment_detected_objects.py` |
| `data/raw/combined_traffic_stats_Thailand.geojson`, `data/raw/ADBMaha_combined_20250808.geojson` | TomTom Traffic Stats route analyses | TomTom Traffic Stats, under your own licence |
| `data/processed/tomtom_segments_{country}.parquet` | the TomTom routes matched to segments | `python src/tomtom_stats.py` |
| `data/processed/adb_segments_tomtom_{country}.parquet` | the ADB segments enriched with measured speeds | `python src/tomtom_data_integration.py` |
| `data/external/tha_ppp_2020.tif`, `data/external/maharashtra_ppp_2020.tif` | WorldPop population rasters, about 2 GB | `python -c "import sys; sys.path.insert(0, 'src'); from fetch_worldpop import fetch_thailand, fetch_maharashtra; fetch_thailand(); fetch_maharashtra()"` |
| `data/external/thailand-260621.osm.pbf`, `data/external/western-zone-260621.osm.pbf` | OpenStreetMap extracts | Geofabrik; `src/prefilter_pbf.py` cuts them down (needs `osmium-tool`) |
| `data/processed/osm_roads_*.parquet`, `osm_crossings_at_grade_*.parquet`, `osm_pedestrian_ways_*.parquet`, `data/processed/osm_vru_layers/` | the OSM road, crossing, pedestrian and VRU layers | `python src/exposure_signals.py && python src/road_separation.py && python src/extract_osm_vru_layers.py` |
| `data/interim/osm_access_{country}.parquet` | legal access per travel mode, read by `src/road_access_join.py` | `python src/road_access.py` |
| `data/processed/isochrones/*.parquet`, `data/processed/poi_zones/` | school and hospital walking isochrones | `python src/poi_isochrone.py --country thailand --poi-type school --force`, with the Valhalla container from `valhalla/README.md` |

`src/pipeline_steps.py` checks many of these paths before its step runs, so a Full or
Complete run in the browser panel names the file it is missing and moves on. The OSM
layers are the exception: the build steps read them directly, so a Full or Complete
run stops at `stage1.road_structure` until
`data/processed/osm_roads_{country}.parquet`,
`osm_crossings_at_grade_{country}.parquet`,
`osm_pedestrian_ways_{country}.parquet`, `osm_vru_layers/{country}/` and
`data/interim/osm_access_{country}.parquet` are in place. The Quick preset uses none
of them.

## Full rebuild

With the files above in place:

```bash
cd /Users/kentaroiio/Documents/github/5s-phase-2
python src/build_v_safe.py     # rewrites data/processed/segments_v_safe.parquet
python src/quick_reproduce.py  # the deliverables
python src/build_tiles.py      # docs/segments_priority.pmtiles (needs tippecanoe)
```

PIPELINE.md §8 gives the same sequence with the optional refresh steps.
README.md describes the method and the results.
