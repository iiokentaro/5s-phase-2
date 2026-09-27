# Physical separation indicated in OSM data

`src/extract_osm_vru_layers.py` reads one `.osm.pbf` file and writes five GeoParquet files (EPSG:4326) to `{out_dir}/{region}/`. Together they show where OpenStreetMap records separation between motor traffic and vulnerable road users (VRUs: pedestrians and cyclists).

```
cd /Users/kentaroiio/Documents/github/5s
python src/extract_osm_vru_layers.py data/external/thailand-260621.osm.pbf data/processed/osm_vru_layers --region thailand
```

## Files

| Parquet file | Extraction rule / target tags | Role in spatial analysis |
| --- | --- | --- |
| `vru-ways.parquet` | Ways with `highway` = `footway`, `cycleway`, `pedestrian`, `path` or `bridleway`. `footway=*`, `cycleway=*`, `bicycle`, `foot`, `segregated`, `surface` and `access` are kept as columns. | **Pedestrian and cycle ways in general.** Separately mapped footways and cycleways are merged into one layer, so a single spatial search finds a VRU way running parallel or next to a road. |
| `vru-attributes-on-roads.parquet` | Ways with `highway` = `motorway`, `trunk`, `primary`, `secondary`, `tertiary`, `unclassified`, `residential`, their `*_link` variants, `living_street`, `road` or `service` that carry any `sidewalk`, `sidewalk:*`, `cycleway` or `cycleway:*` key (for example `sidewalk=both/left/right`, `cycleway=track/lane`, `cycleway:separation=*`, `cycleway:left:separation=*`). | **VRU facilities recorded on the road itself.** Shows sidewalks and cycle tracks or lanes that are tagged on the carriageway without being drawn as separate ways. |
| `barriers.parquet` | Nodes and ways with `barrier` = `guard_rail`, `fence`, `wall`, `kerb`, `bollard` or `hedge`. | **Physical barriers.** A buffer search tests whether a barrier lies between a road and a VRU way. |
| `crossings.parquet` | Ways with `footway=crossing` or any `crossing=*` tag; nodes with `highway=crossing` that belong to none of those ways. | **Crossings.** Points where VRUs cross the carriageway and the separation from motor traffic is interrupted. |
| `intersections-and-controls.parquet` | Nodes with `highway=traffic_signals` or `highway=stop`, and at-grade intersection nodes of the roads listed above (without `service`). | **Intersections and control points.** Locations where motor traffic and VRUs meet, with the type of traffic control. |

## Columns

Every file has `osm_type` (`node` or `way`), `osm_id`, one column for each main tag, `tags` (all tags of the object as a JSON string) and `geometry`.

`intersections-and-controls.parquet` has these columns in place of tag columns:

| Column | Meaning |
| --- | --- |
| `control` | `traffic_signals`, `stop`, or empty for an intersection with no control tag |
| `is_intersection` | True when the node is an at-grade intersection |
| `degree` | Number of road edges that meet at the node |
| `way_count` | Number of road ways that pass through the node |
| `road_highways` | Sorted `highway` values of those road ways |

## Notes

- **Intersection rule.** Each road way adds 1 for a node at either of its ends and 2 for a node in its middle. A node whose total is 3 or more, across 2 or more ways, is an intersection. Two ways joined end to end total 2 and count as a split point. A bridge shares no node with the road below it in OSM, so every intersection found this way is at grade.
- **Crossings drawn as ways.** When a crossing is mapped as a way, its `highway=crossing` nodes are dropped, so each crossing appears once, as the way.
- **Service roads.** `highway=service` is included in `vru-attributes-on-roads.parquet` and left out of the intersection count, because parking aisles and driveways would otherwise turn every access point into an intersection.
- **Geometry types.** Nodes are Points and ways are LineStrings. Closed ways tagged `area=yes`, such as pedestrian plazas and walled enclosures, are MultiPolygons, because osmium always builds them as areas.
- **Overwriting.** File names carry no timestamp, so a new run for the same region overwrites the previous files.

## Row counts (extracts dated 2026-06-21)

| Parquet file | `thailand` | `maharashtra` (`western-zone-260621.osm.pbf`) |
| --- | ---: | ---: |
| `vru-ways.parquet` | 99,035 | 30,604 |
| `vru-attributes-on-roads.parquet` | 10,919 | 3,468 |
| `barriers.parquet` | 22,552 | 10,141 |
| `crossings.parquet` | 21,618 | 1,809 |
| `intersections-and-controls.parquet` | 1,737,334 | 1,494,294 |
