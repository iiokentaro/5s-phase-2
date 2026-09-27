<p align="center">
  <img src="5S%20Logo.png" alt="5S logo" width="240">
</p>

# 5S (Safe System Speed Scoring System)

A submission for the Asian Development Bank's AI for Safer Roads, Safer Speeds Challenge.

## Submission deliverables

| Deliverable | Description | Where to find it |
|---|---|---|
| Analytical model | Code, method and evaluation | [`src/`](src/), [`PIPELINE.md`](PIPELINE.md), and the Method section below |
| Speed Safety Score | Per-segment score (0-100), priority class and review track | [`data/processed/segments_v_safe.parquet`](data/processed/segments_v_safe.parquet); CSV lists in [`outputs/`](outputs/); plain-language guide [`docs/speed_safety_score.md`](docs/speed_safety_score.md) |
| Geospatial visualization | Interactive map of the segments whose speed limit should be reviewed | `python src/serve_map.py` ([`docs/index.html`](docs/index.html) + [`docs/segments_priority.pmtiles`](docs/segments_priority.pmtiles)); GIS layers `outputs/segments_priority.parquet`, `outputs/segments_priority.gpkg` and `outputs/segments_priority.gdb.zip` |

**Philosophy**
* **i. Measure what is measurable, and make measurable what is not so.**
* **ii. Avoid compounding vague assumptions to prevent deliverables from becoming mere artifacts.**
* **iii. Unless there is sufficient evidence to justify raising the speed limit above 30 km/h (including cases with missing data), the Safe Speed shall be set to 30 km/h.**

## Overview

The question: is the posted speed limit (`SpeedLimit`) on each road segment in line with the Safe System, that is, with the speed at which a person survives being hit? 5S answers it per segment and lists the segments whose limit should be reviewed first.

1. For each segment, 5S sets a safe speed, `V_safe`, from how the road is built and who may legally use it (`src/safe_speed.py`).
2. It computes `misalignment = SpeedLimit − V_safe`. A positive value means the sign allows more than the safe speed.
3. It ranks segments by a Speed Safety Score that combines the misalignment with VRU exposure and data confidence (VRU: vulnerable road users, meaning pedestrians, cyclists and motorcyclists).

The ranking is the deliverable. `V_safe` is the yardstick behind it.

`V_safe` never reads the posted limit, and reads measured speed in one place only: the Overture motorway fallback for access control (see "How V_safe is set"). The organizers' Methodology Warning explains why: a high measured 85th-percentile speed is exactly what a road built for high speed produces. Measured speed is otherwise used to check whether the limit record is believable, and for the diagnostic `operating_gap` (F85 − V_safe).

**Scope.** The ADB data covers 55,884 Thailand segments and 14,082 Maharashtra segments. Speed data exists for 15,121 of them (21.6%), and these form the analysis population. The pipeline cuts them into smaller pieces (see Method), so the output `segments_v_safe.parquet` has 102,508 rows: 75,830 Thailand and 26,678 Maharashtra. 1,257 Thailand rows are flagged `data_quality_flag='invalid_speed'`, which leaves 101,251 valid rows. Total length is 60,617 km for Thailand and 40,266 km for Maharashtra. `overture_segment_id` links every row back to its ADB segment.

A row is a piece of a segment, so a share counted by rows gives a 100 m piece the same weight as a 5 km one. Where that matters, the share by length is given too.

## Results by region

The same pipeline and parameters ran on both regions, with no country-specific tuning.

| | Thailand | Maharashtra |
|---|---|---|
| Priority rows (Top Priority, Priority, Watch) | 14,318 | 4,023 |
| Their length | 11,950 km | 8,052 km |
| ADB segments they come from | 2,556 | 1,003 |
| Mean misalignment of priority rows | +55.9 km/h | +33.1 km/h |
| Rural share of priority rows | 48.1% | 75.1% |
| Review Needed / Field Verification Needed | 14,081 / 237 | 2,770 / 1,253 |
| Field Verification share | 1.7% | 31.1% |

- **Thailand.** Posted limits sit far above Safe System speeds, in rural and urban areas alike.
- **Maharashtra.** The gap is about half of Thailand's. More of the priority rows are rural, where exposure is uncertain and the safety-side correction raises it. The score gives 0.35 of its weight to exposure and 0.15 to confidence, so a segment with a small gap can still rank high. `score_explanation` says why each segment ranked where it did. Nearly one in three Maharashtra limit records needs a site check first.

## How to run

All commands run from the repository root. Setup, once:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Re-extracting OSM data also needs `osmium-tool` (`brew install osmium-tool` on macOS, `apt install osmium-tool` on Debian/Ubuntu).

### In the browser

```bash
python src/serve_map.py --enable-pipeline
```

This opens http://localhost:8000/ with a run panel beside the map. Pick a preset and press **Run**. The panel shows each step with its time and log, and reloads the map when new tiles are built.

| Preset | What it runs | Time |
|---|---|---|
| **Quick** (default) | Every deliverable, from the committed `segments_v_safe.parquet` | under a minute |
| **Full** | Missing POI isochrones (with Valhalla), the build from the raw ADB GeoJSON, then every deliverable | minutes; hours if isochrones must be built |
| **Complete** | Everything above, plus WorldPop, the OSM pre-filter and extraction, and Overture Places | hours |

For Full and Complete, the panel also sets the **POI parameters**: the Overture confidence rule, and for each POI type (school, hospital, market, shop, bus stop) whether it caps V_safe, at what speed, and the urban/rural walking minutes of its isochrone. By default only schools apply, at 30 km/h, with 3 / 5 minutes. A run writes the parameters it used to `data/processed/segments_v_safe_poi_params.json`. The build reads the Mapillary data already on disk.

The run API executes repository code. It is off unless you pass `--enable-pipeline`, it listens on 127.0.0.1 only, and it rejects requests from any other `Host` or `Origin`.

### Fast path (about 10 seconds)

```bash
python src/quick_reproduce.py
```

This rebuilds every deliverable from the committed `data/processed/segments_v_safe.parquet`, without the raw GeoJSON (over 130 MB) or the WorldPop rasters (about 2 GB):

- `outputs/priority_map.html`: a folium map. It is too large for GitHub, so it is not committed.
- `outputs/priority_map_static.png`: a static summary.
- `outputs/priority_review_needed.csv`, `outputs/priority_field_check.csv`, `outputs/priority_urban.csv`, `outputs/priority_rural.csv`: the priority lists.
- `outputs/segments_priority.parquet`: GeoParquet for GIS and kepler.gl. The last three columns, `apply_median`, `apply_sidewalk` (boolean) and `bcr` (double), are empty fields to be filled in later.
- `outputs/segments_priority.gpkg`: the same columns as a GeoPackage.
- `outputs/segments_priority.gdb.zip`: the same columns as a zipped File Geodatabase for ArcGIS Online.
- The sensitivity analysis tables, printed to the terminal.

### Full rebuild

```bash
# 1. WorldPop rasters (296 MB for Thailand, 1.8 GB for India)
python -c "import sys; sys.path.insert(0, 'src'); from fetch_worldpop import fetch_thailand, fetch_maharashtra; fetch_thailand(); fetch_maharashtra()"
# 2. The build. It reads data/raw/ADB_Innovation_*.geojson, which this repository
#    holds, and the caches listed in QUICKSTART.md
python src/build_v_safe.py
# 3. The web map's vector tiles (needs tippecanoe)
python src/build_tiles.py
```

`build_tiles.py` puts only the 15 properties the web map reads into `docs/segments_priority.pmtiles`. The full column set stays in the Parquet file. PIPELINE.md §8 lists the steps for rebuilding the OSM extracts, the POI union and the Valhalla isochrones.


### Viewing the map

`python src/serve_map.py` serves the map at http://localhost:8000/. It uses MapLibre GL JS and PMTiles. PMTiles needs HTTP Range requests, so opening `docs/index.html` as a file or with `python -m http.server` does not work. The hosted copy is at https://iiokentaro.github.io/adb-ai-for-safer-roads-safer-speeds-challenge/.

`python src/quick_reproduce.py` writes the same view to `outputs/priority_map_static.png`.

Red = Top Priority, orange = Priority, yellow = Watch, yellow-green = Low Priority, cyan = Aligned. Aligned is a display-only split of `Low Priority`: it marks the Low Priority rows whose limit is already at or below V_safe (`misalignment <= 0`). The rest of Low Priority have a positive gap with a low score.

## Method

PIPELINE.md describes each step, its module and its columns. In build order:

1. **Load and clean** (`src/schema.py`). Rename both countries' columns to one schema. Keep the 15,121 segments with `AnalysisStatus=='Valid'` and a posted limit. Flag the 410 Thailand segments whose three speed fields are all exactly 0 as `invalid_speed`. They stay in the output, and every comparison with the posted limit skips them.
2. **TomTom (optional)** (`src/tomtom_data_integration.py`, `src/tomtom_enrichment.py`). ADB supplied TomTom Traffic Stats for both regions. Segments are cut into uninterrupted stretches between junctions, and each TomTom segment is matched to them by shape. Where TomTom has a posted limit, it replaces the ADB one, and the ADB value stays in `speed_limit_adb`. TomTom also supplies the measured mean and standard deviation for the Elvik estimate, and adds one step of data confidence. For a country without a TomTom file, the build runs on the ADB data alone.
3. **Road structure** (`src/road_separation.py`). Each segment is matched to the OSM ways it was built from, by identical coordinates: 98.35% of Thailand length and 99.20% of Maharashtra length. From those ways come `is_access_controlled` (pedestrians and cyclists may not enter), `is_divided` (opposing traffic cannot meet), `is_grade_separated`, and the legal access per travel mode (`src/road_access_join.py`).
4. **Exposure** (`src/pop_density.py`, `src/exposure_signals.py`, `src/exposure_level.py`). Population density, POI density (OSM plus Overture Places), Mapillary VRU detections and crossings form an exposure level (High / Medium / Low). Urban and rural areas are scored separately. It feeds the priority score only.
5. **V_safe** (`src/safe_speed.py`, `src/junction_speed_cap.py`). Start at 30 km/h. Raise only where the road is access-controlled, then cap with VRU evidence and the 300 m junction cap. See "How V_safe is set" below.
6. **Split around influence zones** (`src/segment_localization.py`). Long segments are cut where a VRU or junction zone begins and ends. Steps 3 to 5 then run again on the pieces.
7. **POI zones, short segments, Mapillary objects** (`src/poi_speed_zones.py`, `src/sandwich_segments.py`, `src/segment_detected_objects.py`). On the final geometry, POI walking isochrones cap V_safe. A short segment between two lower neighbours takes their speed. Roadside objects that Mapillary detected can then raise V_safe on segments outside POI zones.
8. **Misalignment and plausibility** (`src/misalignment.py`, `src/speedlimit_plausibility.py`). Compute `misalignment` and `operating_gap`, and flag limit records that look wrong.
9. **Benefit estimate** (`src/elvik_2019.py`). The expected drop in fatal crashes if speeds came down to V_safe. Step 10 uses it to order segments that share a score or a rank. It never changes a score.
10. **Score, classes and lists** (`src/safety_score.py`, `src/review_track.py`, `src/priority_lists.py`). Compute the Speed Safety Score and the priority class, split the priority rows into Review Needed and Field Verification Needed, and rank them within urban and rural lists.

### How V_safe is set

V_safe starts at 30 km/h on every segment. Only evidence that pedestrians and cyclists cannot be on the road can raise it, and where a motorcyclist may be present, evidence that a barrier separates riders from four-wheeled traffic. Every later rule is a cap applied with `min()`. The full condition table is in [`docs/v_safe_raise_conditions.md`](docs/v_safe_raise_conditions.md).

| Condition | V_safe | `v_safe_basis` |
|---|---|---|
| Default | 30 | `pedestrian:default_vru_possible` |
| Access-controlled and divided, motorcycles prohibited or separated by a barrier | 100 motorway, 90 trunk, 80 other | `separated:*` |
| Access-controlled, undivided, motorcycles prohibited or separated by a barrier | 70 | `head_on:*` |
| Access-controlled, a motorcyclist may be present with no barrier between riders and four-wheeled traffic | 30 | `motorcycle:*` |
| No access control, but a barrier protects pedestrians, cyclists and motorcyclists | 50 | `side_impact:vru_barrier` |
| Cap: Mapillary VRU detection (school-zone sign, crosswalk or bicycle marking) | at most 30 | `pedestrian:vru_detected` |
| Cap: within 300 m of a junction (full motorways and grade-separated segments excluded) | at most 50 | `side_impact:junction_buffer` |
| Cap: inside the walking isochrone of an enabled POI type | at most the type's speed | `pedestrian:poi_zone_<speed>kmh` |

- **Access control.** `highway=motorway`, `motorroad=yes`, or OSM access tags that bar both pedestrians and cyclists. Where OSM shows none of these, an Overture `motorway` with F85 ≥ 50 km/h also counts. This is the only place measured speed touches V_safe.
- **Motorcycles.** The 70-100 km/h rungs price crashes between people inside vehicles. Where a rider may legally share the carriageway with four-wheeled traffic, V_safe stays at 30 with or without a median, because a median separates the two directions and riders still share lanes with cars. A barrier between riders and four-wheeled traffic, read from Mapillary objects (`has_motorcycle_separation`), opens the higher rungs. OSM records no barrier on any matched way.
- **POI zones.** For each enabled type, the zone is the Valhalla walking isochrone around each POI: 3 minutes urban and 5 rural for schools. A zone covering at least 80% of a segment applies to all of it. A zone covering less cuts the segment and applies to the covered pieces only (`is_school_zone`, `is_near_hospital`, ...).
- **Short segments** (`is_sandwich`). A segment up to 250 m long, whose neighbours at both ends all have a lower V_safe, takes the higher of its two along-road neighbours' values.
- **Mapillary objects.** On segments outside POI zones, a detected physical median sets `is_divided` (`mapillary_divided`). Detected protection for pedestrians, cyclists and motorcyclists together sets `has_vru_barrier` (50 km/h without access control). Detected protection for motorcyclists sets `has_motorcycle_separation`, which lets an access-controlled segment take the speed of a road without motorcycles. V_safe only moves up here, and the VRU cap and junction cap still apply.

**Result on the current build.** 101,419 of 102,508 rows are at 30 km/h (98.7% by length). Of those, 18,215 carry a Mapillary VRU detection, and 32,360 rows (11.7% of length) lie in a school zone. Access control is confirmed on 1,379 rows (1.4% of length); on 290 of them a motorcyclist may be present with no barrier, so they stay at 30 (`motorcycle:*`). Above 30, all with motorcycles prohibited: 1,072 rows at 80-100, 5 at 70 and 12 at 50. Mapillary objects mark 375 rows as divided and set no barrier, so they raise no V_safe in this build. The junction cap lowers 12 rows. `misalignment` is positive on 96.5% of valid rows, and F85 exceeds V_safe on 99,017 of 101,251 (97.8%).

The 30 km/h share describes the roads. OSM confirms the structure of almost the whole network, and that structure is open to pedestrians and cyclists nearly everywhere.

### Safe System target speeds

Safe System target speeds are impact speeds a human body survives. The table gives them per road condition, with their sources. V_safe uses the 30, 50 and 70 km/h rows and, for its 80-100 km/h rungs, the last row.

| Road Segment Condition / Environment | Safe System Target Speed | Rationale / Biomechanical Basis | References |
| :--- | :--- | :--- | :--- |
| **Mixed traffic / At-grade VRU conflict areas**<br>*(Urban streets, residential/commercial zones, school zones, roads without pedestrian separation)* | **20–30 km/h** | **Human tolerance without external protection**: At ≤30 km/h, VRU survival rate is ~90% (fatality risk ≤10%). Drops to 20 km/h to prevent severe injury (MAIS 3+) or protect children and the elderly. | • ADB (2024b)<br>• ADB (2024a)<br>• ADB (2021)<br>• Hussain et al. (2019)<br>• ITF (2016)<br>• Turner et al. (2016)<br>• Edvardsson Björnberg et al. (2020) |
| **Roadside rigid hazards present without safety barriers**<br>*(Mid-block sections with run-off-road risk)* | **40 km/h**<br>*(Side impact: 30–40 km/h / Frontal impact: 40 km/h)* | **High impact concentration on rigid objects**: Concentrated forces cause severe vehicle intrusion. Human tolerance is 30–40 km/h for side impacts and 40 km/h for frontal impacts. | • ADB (2024b)<br>• ADB (2018)<br>• Turner et al. (2016)<br>• Woolley et al. (2018)<br>• Edvardsson Björnberg et al. (2020) |
| **At-grade intersections (Vehicle-to-vehicle angle/side impact risk)**<br>*(Intersections without angle deflection like roundabouts)* | **50 km/h** | **Vehicle side-impact structural limit**: Limited side crumple zones restrict occupant survival (≤10% fatal risk) in 90° collisions to 50 km/h. *(Overrides to 20–30 km/h if VRUs cross).* | • ADB (2024b)<br>• ADB (2024a)<br>• ITF (2016)<br>• Turner et al. (2016)<br>• Woolley et al. (2018)<br>• Edvardsson Björnberg et al. (2020) |
| **Undivided roads without median barriers**<br>*(Two-lane mid-block sections with head-on crash risk)* | **70 km/h** | **Vehicle frontal-impact structural limit**: Front crumple zones and airbags protect occupants in head-on collisions between equal-mass cars up to 70 km/h. | • ADB (2024b)<br>• ADB (2024a)<br>• Doecke et al. (2020)<br>• ITF (2016)<br>• Turner et al. (2016)<br>• Edvardsson Björnberg et al. (2020) |
| **High-standard roads with major crash risks physically eliminated**<br>*(Divided roads with medians, grade-separated, VRUs excluded)* | **≥100 km/h** | **Structural elimination of key crash types**: Head-on, side-impact, and VRU conflicts are physically eliminated via median barriers, grade separation, and access control. | • ADB (2024b)<br>• ITF (2016)<br>• Woolley et al. (2018)<br>• Edvardsson Björnberg et al. (2020) |

### Schools count twice, on purpose

A school puts children near the road. It caps V_safe (the isochrone zone), and it also raises the exposure level (`osm_poi_category_count`). Two reasons support this double weight. Children judge traffic less well than adults (Whitebread and Neilson 2000; Plumert, Kearney, and Cremer 2004; Barton and Schwebel 2007; Schwebel, Davis, and O'Neal 2012). A death at a young age also costs the most years of life (Murray and Lopez 1996; World Health Organization 2018).

### AADT estimation

The deliverables carry `AADT`, annual average daily traffic in vehicles per day, estimated from the probe data (`src/aadt_estimation.py`) because no count layer covers the analysis population.

`sample_size_avg`, the mean probe count per sampling point of a segment, is converted to a volume with one scale factor per region. The factors are calibrated in [`docs/AADT Scaling.xlsx`](docs/AADT%20Scaling.xlsx) against the seven segments whose traffic was counted (four in Maharashtra, Egis India Consulting Engineers 2012; three in Thailand, Department of Rural Roads 2026, 2025), with each count carried to the probe year by GDP growth (World Bank `NY.GDP.MKTP.KD.ZG`). PIPELINE.md, Phase 4, gives the calibration step by step.

```
AADT = clip(round(sample_size_avg × weighted_scale × scaling_to_2025), 10, 200000)
```

| Region | `weighted_scale` | `scaling_to_2025` |
|---|---|---|
| Maharashtra | 1.09533 | 1.0 |
| Thailand | 0.04500 | 1.02442 |

Every piece of a split segment keeps its parent's `sample_size_avg`, and so its parent's estimate. Probe counts span seven orders of magnitude while the calibration has seven points, so the result is held between 10 and 200,000 vehicles/day: 9,284 of 102,508 rows sit at the ceiling, 156 at the floor. The browser panel takes both factors as inputs.

## The Speed Safety Score

```
safety_score = 100 × (0.50 × misalignment_score + 0.35 × exposure_score + 0.15 × confidence_score)
```

| Axis | Scaled to 0-1 | Why it is there |
|---|---|---|
| (a) Misalignment, 0.50 | `misalignment_magnitude` / 60, capped at 1 | The question itself: where a limit review should start |
| (b) VRU exposure, 0.35 | Low 0, Medium 0.5, High 1 | At the same gap, priority goes where people are |
| (c) Data confidence, 0.15 | High 1, Medium 0.67, Low 0.33 | A list topped by doubtful records would be dismissed |

Confidence counts how many of three columns read `low`: exposure confidence, road-structure confidence and limit plausibility. None gives High, one gives Medium, two or more give Low. A row matched to TomTom gets +1/3, capped at 1. The weights are the team's choice, and the sensitivity analysis below tests them. Confidence has the smallest weight: the Safe System errs on the side of safety, so doubtful data should not push a dangerous segment off the list.

Every row carries `score_explanation`, a plain sentence such as "Posted speed limit is 40 km/h above the safe speed (V_safe). VRU exposure: Medium. Data confidence: High." It also names the speed data source.

### Priority classes

Each of the four (country, land_use) cells is sorted by score, and its top 3%, 10% and 20% of road length form the classes. A row joins a class when the length ranked above it is below the class's share, so the row that crosses a boundary is included. Length is used because the build splits segments around POI and junction zones, and a count would give a heavily split stretch more of the list. Rows with the same score are ordered by `exp_delta_fatal_percent_uniform` (largest first, missing last), then by `segment_id`. Thailand and Maharashtra are separate programmes, and urban and rural exposure are scored on different scales. A pooled cut would let one group crowd out the other.

| Class | Rule (share of the cell's length) | Thailand | Maharashtra | Total |
|---|---|---|---|---|
| Top Priority | top 3% | 1,849 rows, 1,794 km | 627 rows, 1,210 km | 2,476 rows, 3,004 km |
| Priority | top 3-10% | 5,175 rows, 4,181 km | 1,430 rows, 2,839 km | 6,605 rows, 7,019 km |
| Watch | top 10-20% | 7,294 rows, 5,975 km | 1,966 rows, 4,004 km | 9,260 rows, 9,979 km |
| Low Priority | the rest | 60,255 rows, 47,788 km | 22,655 rows, 32,206 km | 82,910 rows, 79,994 km |

Many rows sit at V_safe = 30 with the gap capped at 60 km/h, so scores bunch on a few values: 13.2% of Thailand rural rows score 100. A cut at a score value would put every tied row in the higher class, leaving Thailand rural with no Priority and Thailand urban with no Watch. The Elvik tie-break decides which of the tied rows fill a class. It never changes a score.

### Review Needed and Field Verification Needed

The priority rows (Top Priority, Priority, Watch) are split by whether the limit record looks believable (`src/speedlimit_plausibility.py`). A record is flagged `low` if it is an IQR outlier within its (country, road_class, land_use) group (6,934 rows), or if it differs from F85 by more than 30 km/h (10,579 rows). In total, 14,753 of 101,251 valid rows (14.6%) are flagged.

- **Review Needed** (16,851 rows): the record is believable, so the gap is a real candidate for lowering the limit.
- **Field Verification Needed** (1,490 rows): the record looks wrong, so the sign should be checked on site first.

No official speed-limit dataset exists to check against, so the two lists are kept apart. Mixing likely data errors into the review list would erode trust in it.

### Urban and rural lists

`outputs/priority_urban.csv` and `outputs/priority_rural.csv` split the priority rows by `road_environment`: `rural` is `land_use=='RURAL'` plus every motorway, and `urban` is the rest. `rank_within_environment` uses misalignment and exposure only, in the same 0.50 : 0.35 ratio. Rural rows raised by the safety-side correction have low confidence by design, and leaving confidence out keeps them from being pushed down. `confidence_note` still records it. Rows that share a rank are ordered by `exp_delta_fatal_percent_uniform`, largest first.

`exp_delta_fatal_abs = exp_delta_fatal_percent_uniform / 100 × sample_size_avg` weights the benefit by probe count, a proxy for traffic volume. It lets urban and rural rows be compared on one scale. The data holds no crash counts, so it is a relative measure and makes no claim about lives. The top row is a Thailand rural motorway segment (90.3 million samples, 98.8%). Urban rows have far more samples (median 1,158,130 against 72,434 for rural), so this measure favours urban roads.

## Fatal-crash reduction estimate (Elvik 2019)

For each segment, `src/elvik_2019.py` estimates how much fatal crashes would drop if the speed distribution moved down until its mean reached V_safe (Elvik 2019, `docs/Elvik (2019).pdf`; coefficients in `src/elvik_2019.json`). The reported column is `exp_delta_fatal_percent_uniform`.

- Relative injury rate `= exp(k × (v − v_ref))`, with k = 0.08 for fatal, 0.06 for serious and 0.04 for slight. The reported figure uses the fatal value.
- Speeds are taken as normally distributed and split into 12 half-SD intervals over the mean ± 3 SD, as in the paper's Table 1.
- The mean and SD come from TomTom where both are measured. Elsewhere they come from ADB, with `median_speed` as the mean and `(F85 − mean) / 1.04` as the SD. `exp_speed_source` records which was used.
- `uniform` (reported) shifts the whole distribution. `tailcap` caps only the speeds above V_safe and is kept as a sensitivity check.
- `src/elvik_2019.py` reproduces the paper's Table 2 (38.1%) and Table 3 (29%).

Results on the Review Needed rows with a positive estimate (`median_speed > v_safe`: 16,584 of 16,851, or 98.4%):

| Road environment | n | Mean `uniform` (reported) | Mean `tailcap` |
|---|---|---|---|
| rural | 9,092 | 90.3% | 94.1% |
| urban | 7,492 | 90.1% | 94.7% |

The other 267 rows already move at or below V_safe. Their value is zero or negative, is read only as a sign, and is left out of the table. Eight valid rows have no usable spread (F85 ≤ median) and get NaN.

Limitations:
- (a) Real speed distributions are close to normal, but not exactly. On TomTom rows, the measured 19-point distribution gives nearly the same `uniform` value (rural 83.9% vs 83.8%, urban 88.9% both ways) and a `tailcap` value 1.1 to 1.2 points lower.
- (b) k has no published confidence interval, so the figure carries no range.
- (c) If a measure acts mainly on the fastest drivers, the higher `tailcap` figure applies.
- (d) The figure assumes operating speed actually reaches V_safe. A new sign alone may not achieve this, especially where `operating_gap` is large.

## Sensitivity analysis

`src/sensitivity_analysis.py` tests how much the Top Priority list depends on sample size and on the weights, and how the benefit estimate depends on its assumptions. It reuses `safety_score.add_safety_score` unchanged.

**(1) Sample size.** Remove the bottom 25% of rows by `sample_size_avg` (fewer than 22,450 samples; 25,302 rows) and recompute. The removed rows are long ones: they carry 44% of the length (56,126 of 99,996 km remain). The baseline Top Priority list has 2,476 rows and 3,004 km, and 102 of those rows (201 km) fall in that bottom quarter. The recomputed list is 3% of the remaining length, so it is shorter: 1,747 rows and 1,687 km, all of them in the baseline list (recall 56.2% by length). No segment outside the baseline list enters when the thin quarter is removed, so the list is not driven by thinly sampled segments.

**(2) Weights.**

| Weights (misalignment / exposure / confidence) | Top Priority | Overlap | Recall | Jaccard |
|---|---|---|---|---|
| baseline 0.50 / 0.35 / 0.15 | 3,004 km | 3,004 km | 100.0% | 100.0% |
| misalignment-heavy 0.70 / 0.20 / 0.10 | 3,004 km | 3,004 km | 100.0% | 100.0% |
| exposure-heavy 0.30 / 0.55 / 0.15 | 3,007 km | 2,998 km | 99.8% | 99.5% |
| confidence-heavy 0.40 / 0.30 / 0.30 | 3,003 km | 2,817 km | 93.8% | 88.3% |
| equal 0.33 / 0.33 / 0.34 | 3,003 km | 2,817 km | 93.8% | 88.3% |

Recall and Jaccard are measured by length. Every scenario keeps at least 93.8% of the baseline list's length. The list holds 3% of each cell's length in every scenario, so recall and Jaccard move together.

**(3) Benefit estimate.** Among the Review Needed rows above, `tailcap` exceeds `uniform` by 3.8 points on rural roads and by 4.6 on urban roads. The highest speeds carry the highest risk, so a measure aimed at the fastest drivers is worth more where speeds vary most. The measured TomTom distribution moves `uniform` by at most 0.1 points. Neither choice changes a rank. To report all three severities, run `python src/sensitivity_analysis.py --severity all3`.

## Known limitations

- **Coverage.** Only the 15,121 segments with speed data (21.6% of the network) are analysed. The method says nothing about the rest.
- **The input attributes are estimates.** `SpeedLimit`, `LandUse` and `RoadClass` come from Overture, and no official limit dataset exists to check them (the organizers' FAQ says the same). The Field Verification list needs a site visit to settle.
- **Time lag.** The road network dates from December 2024 (Thailand) and May 2025 (Maharashtra). OSM data is from June 2026 and Overture Places from August 2026.
- **VRU exposure is indirect.** No pedestrian or cyclist counts exist. Rural OSM and Mapillary coverage is close to zero (0 to 4 pedestrian tags and no images in the rural samples checked), so rural exposure often rests on population density alone. A safety-side correction raises rural exposure to at least Medium where population density is in the top quarter for rural roads of that country and no crossing was found. This applies to 25.4% of rural Thailand rows and 30.3% of rural Maharashtra rows.
- **V_safe depends on OSM tags.** Access control, division and motorcycle access come from OSM. 3,339 rows have `road_structure_confidence='low'`, and they stay at 30 km/h unless the Overture motorway fallback applies.
- **Grade separation is flagged broadly.** A segment counts as grade-separated if any matched way is a bridge, tunnel or non-zero layer, which OSM's many short canal-bridge ways make true for 24.7% of Thailand rows and 25.2% of Maharashtra rows. Those rows are excluded from the 50 km/h junction cap, so an access-controlled segment that meets an at-grade junction can keep 70 to 100 km/h because a bridge lies somewhere on it.
- **Weights and thresholds are choices.** The score weights, the class cut-offs (3% / 10% / 20%), the isochrone minutes, the 300 m junction radius and the 250 m sandwich length are set by the team. All are parameters for road authorities to retune to their budget and staffing.
- **Speed data is copied down to the pieces.** A piece inherits its parent's posted limit, observed speeds, `sample_size_avg` and `sample_size_total`. Speed may still vary within a segment, and summing `exp_delta_fatal_abs` over pieces counts the parent more than once.
- **One TomTom segment can speak for a long stretch.** TomTom values fill the whole uninterrupted stretch they match. `tomtom_covered_len_m_*` shows how much of each piece TomTom actually measured.
- **`AADT` is scaled from probe counts, calibrated on seven segments.** Four counted segments in Maharashtra and three in Thailand set one factor per region (Pearson's r between counted volume and probe count: 0.95 in Maharashtra, 1.00 in Thailand). Two of Maharashtra's four carry the same counted volume (20,458 vehicles/day in 2012). The estimate assumes probe counts are proportional to traffic and that GDP growth tracks traffic growth. It is a screening value: 9% of rows sit at the 200,000 ceiling, and a real count layer should replace it.
- **Junction nodes are a manual export.** `data/external/osm_junctions_*.geojson` came from Overpass Turbo (`highway=traffic_signals` or `junction=yes`). To refresh it, re-run that query.
- **Use it as a screening tool.** Without crash records or ground truth, external validity is untested. The output shows where to measure and review first. Direct measurement of speeds, VRU counts and posted limits would remove many of the assumptions above.

## Data sources and licences

`src/auxiliary_data.py`. Every source except Mapillary is free, needs no login, and is read from a fixed URL.

| Data | Source | Licence | Use |
|---|---|---|---|
| Road segments, posted limits, speeds | Overture Maps, supplied by ADB (`data/raw/`) | ODbL | Analysis population |
| TomTom Traffic Stats (optional) | Supplied by ADB (`data/raw/combined_traffic_stats_Thailand.geojson`, `data/raw/ADBMaha_combined_20250808.geojson`) | Supplied for this challenge | Finer geometry, posted limits, speed distribution, confidence |
| Road attributes and POIs | OpenStreetMap, Geofabrik extracts of 2026-06-21 | ODbL | Access control, division, grade separation, POIs, crossings |
| POIs | Overture Maps Places, release `2026-08-19.0` (`src/fetch_overture_pois.py`) | CDLA Permissive 2.0 | Joined with the OSM POIs |
| Population density | [WorldPop](https://hub.worldpop.org/) 2020, 3 arc-seconds | CC BY 4.0 | Exposure |
| Street-image detections | [Mapillary](https://www.mapillary.com/) Graph API v4 and vector tiles | CC BY-SA 4.0 | VRU evidence (26 classes of school-zone signs, crosswalk markings and bicycle markings, listed in `src/fetch_mapillary_features.py`) and roadside objects |
| Traffic counts, seven segments | Maharashtra: NH-211 project report (Egis India Consulting Engineers 2012). Thailand: rural road network AADT (Department of Rural Roads 2026, 2025). Worked up in `docs/AADT Scaling.xlsx` | Public government reports | Calibrating the AADT scale factors |
| GDP growth (annual %) | [World Bank](https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG) `NY.GDP.MKTP.KD.ZG`, India and Thailand | CC BY 4.0 | Carrying counted traffic to the probe year in the AADT calibration |

The POI union (`src/poi_sources.py`) keeps an Overture place when its `confidence` is at least 0.5 or it is in the top 95% of its country and category. An Overture point within 10 m of a same-category OSM feature counts as the same place. With the default settings this gives 53,041 schools for Thailand and 15,803 for Maharashtra.

Mapillary needs a personal token (`MAPILLARY_TOKEN` in `.env`, which is excluded from the repository). See `data/mapillary/note.md`.

Attribution: "Contains OpenStreetMap data, © OpenStreetMap contributors, ODbL" / "Contains Overture Maps Foundation data, ODbL" / "Contains Overture Maps Foundation Places data, CDLA Permissive 2.0" / "Mapillary imagery © Mapillary contributors, CC BY-SA 4.0" / "WorldPop population data, CC BY 4.0" / "World Bank Open Data, CC BY 4.0".

## References

Asian Development Bank. 2018. *CAREC Road Safety Engineering Manual 3: Roadside Hazard Management*. Manila: Asian Development Bank. http://dx.doi.org/10.22617/TIM179174-2.

Asian Development Bank. 2021. *CAREC Road Safety Engineering Manual 4: Pedestrian Safety*. Manila: Asian Development Bank. https://www.adb.org/publications/carec-road-safety-engineering-manual-4-pedestrian-safety. 

Asian Development Bank. 2024a. *CAREC Road Safety Engineering Manual 6: Identifying, Investigating, and Treating Blackspots*. Manila: Asian Development Bank. https://www.adb.org/publications/carec-road-safety-engineering-manual-6-blackspots.

Asian Development Bank. 2024b. *CAREC Road Safety Engineering Manual 7: Why and How to Manage Speed*. Manila: Asian Development Bank. http://dx.doi.org/10.22617/TIM240367-2.

Barton, Benjamin K., and David C. Schwebel. 2007. "The Roles of Age, Gender, Inhibitory Control, and Parental Supervision in Children's Pedestrian Safety." *Journal of Pediatric Psychology* 32 (5): 517–526. https://doi.org/10.1093/jpepsy/jsm014

Department of Rural Roads (Thailand). 2026. "ปริมาณจราจร" [Traffic Volume (AADT) on the Rural Road Network]. MOT Data Catalog. https://datagov.mot.go.th/dataset/aadt1.

Department of Rural Roads (Thailand), Bureau of Maintenance (สำนักบำรุงทาง). 2025. บัญชีโครงข่ายทางหลวงชนบท ประจำปีงบประมาณ 2570 [Rural Road Network Inventory, Fiscal Year 2570 (2026/27)]. Bangkok: Department of Rural Roads. https://datagov.mot.go.th/dataset/dataset_12_011.

Doecke, Sam D., Matthew R. J. Baldock, Craig N. Kloeden, and Jeffrey K. Dutschke. 2020. "Impact Speed and the Risk of Serious Injury in Vehicle Crashes." *Accident Analysis & Prevention* 144: 105629. https://doi.org/10.1016/j.aap.2020.105629.

Edvardsson Björnberg, Karin, Mikael Belin, and Claes Tingvall, eds. 2020. *The Vision Zero Handbook*. Cham: Springer. https://doi.org/10.1007/978-3-030-23176-7.

Egis India Consulting Engineers. 2012. Detailed Project Report (DPR), Final: Aurangabad (km 290.2)–Dhule (km 452.8) Excluding Autram Ghat (km 376 to km 390) Section of NH-211 in Maharashtra State, Volume I, Main Report. Report Code BI 00 072. New Delhi: National Highways Authority of India. https://environmentclearance.nic.in/writereaddata/Online/TOR/0_0_29_Jul_2014_1213560671REPORT.pdf.

Elvik, Rune. 2019. "A comprehensive and unified framework for analysing the effects on injuries of measures influencing speed." *Accident Analysis & Prevention* 125: 63–69. https://doi.org/10.1016/j.aap.2019.01.033

Hussain, Qinaat, Hanqiu Feng, Raphael Grzebieta, Tom Brijs, and Jake Olivier. 2019. "The Relationship Between Impact Speed and the Probability of Pedestrian Fatality During a Vehicle-Pedestrian Crash: A Systematic Review and Meta-Analysis." *Accident Analysis & Prevention* 129: 241–249. https://doi.org/10.1016/j.aap.2019.05.033.

International Road Assessment Programme. 2026. iRAP Methodology Reference Guide. Version 3.10. London: International Road Assessment Programme. https://irap.org/ High-volume/methodology-reference-guide.

International Transport Forum. 2016. *Zero Road Deaths and Serious Injuries: Leading a Paradigm Shift to a Safe System*. Paris: OECD Publishing. https://doi.org/10.1787/9789282108055-en.

Murray, Christopher J. L., and Alan D. Lopez, eds. 1996. *The Global Burden of Disease.* Cambridge, MA: Harvard University Press.

Plumert, Jodie M., Joseph K. Kearney, and James F. Cremer. 2004. "Children's Perception of Gap Affordances: Bicycling Across Traffic-Filled Intersections in an Immersive Virtual Environment." *Child Development* 75 (4): 1243–1253. https://doi.org/10.1111/j.1467-8624.2004.00736.x

Schwebel, David C., Aaron L. Davis, and Elizabeth E. O'Neal. 2012. "Child Pedestrian Injury: A Review of Behavioral Risks and Preventive Strategies." *American Journal of Lifestyle Medicine* 6 (4): 292–302. https://doi.org/10.1177/0885066611404876

Turner, Blair, Chris Jurewicz, Kate Pratt, Bruce Corben, and Jeremy Woolley. 2016. *Safe System Assessment Framework*. Research Report AP-R509-16. Sydney: Austroads.

Whitebread, David, and Kevin Neilson. 2000. "The Contribution of Visual Search Strategies to the Development of Pedestrian Skills by 4–11 Year-Old Children." *British Journal of Educational Psychology* 70 (4): 539–557. https://doi.org/10.1348/000709900158290

Woolley, Jeremy, Chris Stokes, Blair Turner, and Chris Jurewicz. 2018. *Towards Safe System Infrastructure: A Compendium of Current Knowledge*. Research Report AP-R560-18. Sydney: Austroads.

World Bank. 2026. "GDP Growth (Annual %)." World Bank Open Data. https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG.

World Health Organization. 2018. *Global Status Report on Road Safety 2018.* Geneva: World Health Organization. https://iris.who.int/server/api/core/bitstreams/9c866a4e-fda7-43bd-96df-27d7c3b509bc/content

## Declaration of AI tool use

Anthropic's Claude Opus 5 was used for coding assistance and language editing on this repository. The team remains responsible for the content of the repository.
