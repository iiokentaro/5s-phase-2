# Conditions for raising V_safe above 30 km/h

Every segment starts at V_safe = 30 km/h. This document states, per travel mode, every condition a segment must meet for the pipeline to give it a higher value, and the speed each combination yields. The rules are identical in Maharashtra and Thailand; §5 and §6 record what differs between the two (default access values, the statutes behind them, and how many segments each rule reaches).

The rules are implemented in `src/safe_speed.py` (`classify_collision_type`, `compute_v_safe`) and `src/junction_speed_cap.py`, from inputs built by `src/road_structure_flags.py`, `src/road_separation.py`, `src/road_access_join.py` and `src/exposure_signals.py`. §8 records how the table was checked against the pipeline's output.

## 1. How to read this document

1. Logical operators are written as the capitalised words `AND`, `OR`, `NOT`.
2. Every expression that combines two different operators is fully parenthesised. No operator precedence is assumed anywhere, so `(NOT M) OR S` and `NOT (M OR S)` are the only two readings a reader can reach, and the parentheses say which one is meant.
3. `OR` is inclusive: `X OR Y` is true when X is true, when Y is true, and when both are true.
4. **Within one row of the rule table (§4), all cells must be true at the same time.** The cells of a row are joined by `AND`.
5. **Across rows, exactly one row is true for every segment.** The rows are mutually exclusive and together cover every case (proof in §4.4). A segment receives the speed of the one row it satisfies.
6. A cell reading "no condition" is always true.
7. Every variable in §3 is either true or false, and is evaluated separately for each row of the deliverable (an ADB segment, or a part of one after the pipeline splits it).

## 2. Terms used in the definitions

**Matched way, matched length.** `src/segment_way_match.py` attributes each metre of an ADB segment to the OSM way that segment is built from, by coordinate identity of their vertices. A *matched way* of a row is an OSM way with more than 0 m attributed to that row. The row's *matched length* is the total metres attributed to all its matched ways. Metres attributed to no way are *unmatched* and count as evidence for nothing.

**Resolved access value of a way, per mode.** For each OSM way and each mode, `src/road_access.py` reads the way's access tags in this order, from general to specific, and the most specific tag present decides:

| Mode | Tags read, general → specific |
|---|---|
| pedestrian (`foot`) | `access`, `foot` |
| bicycle | `access`, `vehicle`, `bicycle` |
| motorcycle | `access`, `vehicle`, `motor_vehicle`, `motorcycle` |

If none of those tags is on the way, the country's default for the way's `highway=*` value decides (`src/road_access_rules.json`, listed in `docs/road_access_rules.md`). The result is one of `yes`, `restricted`, `no`, `unknown`.

**Resolved access value of a row (`legal_<mode>`).** `src/road_access_join.py` combines the values of the row's matched ways as follows, checking the steps in order and stopping at the first that applies:

1. The row has no matched way → `unknown`.
2. At least one matched way is `yes` → `yes`.
3. At least one matched way is `restricted` → `restricted`.
4. At least one matched way is `no` → `no`.
5. Otherwise → `unknown`.

So `legal_<mode> = no` means: the row has at least one matched way, AND no matched way is `yes`, AND no matched way is `restricted`, AND at least one matched way is `no`.

**Confidence of a row's pedestrian value (`legal_foot_confidence`).** Each way's pedestrian value is `high` if a `foot` tag decided it, `medium` if an `access` tag decided it, and `low` if a country default decided it or the value is `unknown`. The row takes the lowest confidence among its matched ways, and `low` when it has no matched way. So `legal_foot_confidence` is `medium` or `high` exactly when every matched way carries an `access` or `foot` tag that decided its pedestrian value, and that value is other than `unknown`.

## 3. Variables

| Symbol | Pipeline column | True when |
|---|---|---|
| `A_osm` | feeds `is_access_controlled` | The row has at least one matched way, AND every matched way carries `highway=motorway` OR `motorroad=yes` (a way carrying both also qualifies). |
| `A_mw` | feeds `access_control_basis` | The row has at least one matched way, AND every matched way carries `highway=motorway`. (`A_mw` true implies `A_osm` true.) |
| `A_ovt` | feeds `is_access_controlled` | (NOT `A_osm`) AND the ADB segment's Overture `road_class` is `motorway` AND its `f85_speed` is at least 50 km/h. A missing `f85_speed` makes `A_ovt` false. |
| `A_vru` | feeds `is_access_controlled` | `legal_foot = no` AND `legal_bicycle = no` AND `legal_foot_confidence` is `medium` or `high`. |
| `A` | `is_access_controlled` | `A_osm` OR `A_ovt` OR `A_vru`. Meaning: neither a pedestrian nor a cyclist may legally be on the road. |
| `D` | `is_divided` | (The row has at least one matched way, AND at least 95% of its matched length lies on ways satisfying (`lanes:divided=yes` OR `dual_carriageway=yes` OR (`oneway` is present AND its value is none of `no`, `0`, `false`))) OR `mapillary_divided`. `mapillary_divided`: the Mapillary objects detected on a segment outside every POI zone carry a physical-median weight of at least 1 (`src/segment_detected_objects.py`). Meaning: opposing traffic cannot meet. |
| `M` | from `legal_motorcycle` | `legal_motorcycle` is any value other than `no`. So `yes`, `restricted`, `unknown`, and a missing value all make `M` true. Meaning: a motorcyclist may legally be on the road, or it is not known that one may not. |
| `G` | `is_grade_separated` | At least one matched way satisfies (`bridge` is one of `yes`, `boardwalk`, `viaduct`, `movable`, `construction`) OR (`tunnel` is one of `yes`, `building_passage`, `covered`, `culvert`) OR (`layer`, read as a number from its first `;`-separated value, is present AND is not 0). |
| `B_v` | `has_vru_barrier` | The Mapillary objects detected on a segment outside every POI zone carry a protection weight of at least 1 for pedestrians, for cyclists and for motorcyclists (`src/segment_detected_objects.py`). No row has it in the current build. |
| `S` | `has_motorcycle_separation` | The Mapillary objects detected on a segment outside every POI zone carry a protection weight of at least 1 for motorcyclists (`src/segment_detected_objects.py`). Meaning: a physical barrier separates riders from four-wheeled traffic. A median does not count, because it separates the two directions and riders still share lanes with cars. No row has it in the current build: the only scored object with a motorcyclist protection weight, the lane separator, weighs 0.5. |
| `V` | `is_vru` | (`is_mapillary_vru` OR a Mapillary hospital sign, where hospitals cap V_safe) AND (NOT `A`). `is_mapillary_vru`: a Mapillary school-zone sign, crosswalk marking or bicycle marking lies within 200 m (urban) or 400 m (rural). |
| `N` | (not stored) | The segment lies within 300 m (distance at most 300 m) of at least one feature in `data/external/osm_junctions_{country}.geojson` carrying `highway=traffic_signals` OR `junction=yes`. |
| `K` | derived from `access_control_basis` | `A_mw` OR `A_ovt`. Meaning: the road is a full motorway, so any junction nearby is an interchange ramp. |
| `J` | `near_junction` | `N` AND (NOT `K`) AND (NOT `G`). Meaning: the segment passes an at-grade junction. |

The pipeline records which clause of `A` established it in `access_control_basis`, which is used by `K`:

| `access_control_basis` | True when |
|---|---|
| `osm_motorway` | `A_mw` |
| `osm_motorroad` | `A_osm` AND (NOT `A_mw`) |
| `overture_motorway_fallback` | `A_ovt` |
| `osm_vru_prohibited` | `A_vru` AND (NOT `A_osm`) AND (NOT `A_ovt`) |
| empty | NOT `A` |

## 4. The rule table

### 4.1 Table

Columns are travel modes. Within a row, every cell must be true (§1, rule 4). Exactly one row is true for each segment (§1, rule 5).

| Row | V_safe (km/h) | Pedestrian | Bicycle | Motorcycle | Motor-vehicle occupants (car, bus, truck) | `collision_type` written | Reachable today |
|---|---|---|---|---|---|---|---|
| R0 | 30 | The segment satisfies none of R1 to R5. | ← same | ← same | ← same | `pedestrian` | yes |
| R1 | 50 | (NOT `A`) AND `B_v` AND (NOT `V`) | (NOT `A`) AND `B_v` AND (NOT `V`) | no condition | no condition | `side_impact` | yes (no row in the current build) |
| R2 | 30 | `A` | `A` | `M` AND (NOT `S`) | no condition | `motorcycle` | yes |
| R3 | 70 | `A` | `A` | (NOT `M`) OR `S` | (NOT `D`) AND (NOT `J`) | `head_on` | yes |
| R4 | 100 if `road_class` is `motorway`; 90 if `trunk`; 80 for every other `road_class` | `A` | `A` | (NOT `M`) OR `S` | `D` AND (NOT `J`) | `separated` | yes |
| R5 | 50 | `A` | `A` | (NOT `M`) OR `S` | `J` | `side_impact` | yes |

Notes on the table:

- **`A` is one condition covering two modes.** It appears in both the Pedestrian and the Bicycle column because each clause of `A` removes both: `highway=motorway`, `motorroad=yes` and the Overture fallback prohibit both modes at once, and `A_vru` requires both prohibitions. No clause removes only one of the two.
- **`V` does not appear in R2 to R5.** `V` is false whenever `A` is true (§3), so `NOT V` would add nothing to a row that already requires `A`.
- **R2 has no condition on `D` or `J`.** A median separates the two directions of traffic, and riders still share lanes with cars, so `D` does not protect them. The junction cap only lowers V_safe to 50 km/h, which leaves 30 km/h unchanged.
- **R5 is the junction cap:** R3 and R4 become 50 km/h when `J` is true.
- **Two later steps lower V_safe on rows without access control and leave the row unchanged.** Inside the walking isochrone of an enabled POI type, V_safe is capped at the type's speed (`src/poi_speed_zones.py`), and a short segment whose V_safe is higher than every neighbour takes its neighbours' value (`src/sandwich_segments.py`). Both act on R0 and R1 rows only.
- **`S` is false today**, so on today's data `(NOT M) OR S` reduces to `NOT M`, and `M AND (NOT S)` reduces to `M`.

### 4.2 The same rules in words

Each segment gets the speed of the first description below that it satisfies. §4.4 shows that at most one of R1 to R5 can ever apply, so the order changes nothing.

- **R1, 50 km/h.** All of the following hold: pedestrians and cyclists are not legally excluded (`A` false); a pedestrian and cyclist barrier is confirmed (`B_v` true); no VRU evidence was found nearby (`V` false). No row meets it in the current build.
- **R2, 30 km/h.** All of the following hold: pedestrians and cyclists are legally excluded (`A` true); a motorcyclist may be present (`M` true); no barrier separates riders from four-wheeled traffic (`S` false). A median makes no difference.
- **R3, 70 km/h.** All of the following hold: `A` is true; at least one of these holds: motorcyclists are legally excluded (`M` false), or a barrier separates them from four-wheeled traffic (`S` true); the carriageway is undivided (`D` false); the segment passes no at-grade junction (`J` false).
- **R4, 80, 90 or 100 km/h by road class.** All of the following hold: `A` is true; at least one of these holds: `M` is false, or `S` is true; the carriageway is divided (`D` true); the segment passes no at-grade junction (`J` false).
- **R5, 50 km/h.** All of the following hold: `A` is true; at least one of these holds: `M` is false, or `S` is true; the segment passes an at-grade junction (`J` true).
- **R0, 30 km/h.** Every other segment.

### 4.3 Why each condition sits in its column

Each column holds the conditions that remove the crash that mode would otherwise not survive at the higher speed.

| Column | Crash the condition removes | Survivable impact speed it protects |
|---|---|---|
| Pedestrian, Bicycle | A vehicle striking a person on foot or on a bicycle. Removed by legal exclusion (`A`), or by a physical barrier (`B_v`). | 30 km/h |
| Motorcycle | A car striking a rider in the same lane; the rider has no vehicle structure around them. Removed by legal exclusion (`NOT M`), or by a physical barrier between riders and four-wheeled traffic (`S`). A median separates only the two directions of traffic, so it does not remove this crash. | 30 km/h |
| Motor-vehicle occupants | Head-on crashes between vehicles, removed by a divided carriageway (`D`); side impacts at at-grade junctions, removed by passing none (`NOT J`). | 70 km/h head-on, 50 km/h side impact |

### 4.4 Why exactly one row applies

- R1 requires `NOT A`. R2 to R5 all require `A`. So R1 excludes every other row.
- R2 requires `M AND (NOT S)`. R3, R4 and R5 require its negation `(NOT M) OR S`. So R2 excludes R3 to R5.
- R3 and R4 require `NOT J`. R5 requires `J`. So R5 excludes R3 and R4.
- R3 requires `NOT D`; R4 requires `D`. So R3 and R4 exclude each other.
- R0 is defined as "none of R1 to R5", so the rows together cover every segment.

## 5. Maharashtra

**What differs from Thailand.**

- *Default access values.* In the resolved default tables of `docs/road_access_rules.md`, the two countries differ in one cell: motorcycle access on `highway=path` is `unknown` in Maharashtra and `yes` in Thailand. Both values make `M` true. The difference reaches the rule table only through the row combination in §2: a Maharashtra row whose matched ways are a `path` (`unknown`) and a way resolving to `no` resolves to `no`, so `M` is false there, where the same row in Thailand resolves to `yes`. Maharashtra's one override, `track` → `motor_vehicle=yes`, matches Thailand's and does not enter the rule table.
- *Legal basis for the motorway defaults.* `highway=motorway` defaults to `foot=no`, `bicycle=no` and `motorcycle=no`. For Maharashtra's Expressways (Mumbai–Pune Expressway, Samruddhi Mahamarg) and the Rajiv Gandhi Sea Link, those bans are attested by secondary sources only (encyclopedia articles and news reports of MSRDC enforcement and of a state notification); the gazette notifications themselves were not read (`src/road_access_rules.json`, `source.country_law.maharashtra`). In this extract `highway=motorway` also covers Mumbai flyovers and the Sion–Panvel Highway, for which no ban was sourced.
- *Statutory powers.* The Control of National Highways (Land and Traffic) Act 2002, s.35, lets the Highway Administration restrict any class of traffic on **any** National Highway by gazette notification. The Motor Vehicles Act 1988, s.115, lets the State restrict motor vehicles on a specified road. So a National Highway (`highway=trunk`) can carry a pedestrian or bicycle ban that no OSM tag records; the rule table then reads `A` as false, which keeps the segment at 30 km/h.

**Rows reached** (current build; the 26,678 rows come from 3,577 ADB segments, split by the TomTom layer and by the influence-zone and POI-zone steps):

| Row | V_safe | Rows | km |
|---|---|---|---|
| R0 | 30 | 26,264 | 40,182.5 |
| R1 | 50 | 0 | 0 |
| R2 | 30 | 226 | 41.9 |
| R3 | 70 | 0 | 0 |
| R4 | 100 | 188 | 34.4 |
| R5 | 50 | 0 | 0 |
| Total | | 26,678 | 40,258.8 |

Rows with `A` true, by `access_control_basis`: `osm_motorway` 203 (35.1 km; 188 in R4, 15 in R2), `overture_motorway_fallback` 109 (21.2 km; all R2), `osm_motorroad` 89 (17.3 km; all R2), `osm_vru_prohibited` 13 (2.7 km; all R2).

## 6. Thailand

**What differs from Maharashtra.**

- *Default access values.* Motorcycle access on `highway=path` is `yes` (see §5 for the one effect this has). `service` is pinned to `foot=yes` and `motorcycle=yes`, and `track` to `motor_vehicle=yes`; both resolve to the same values as in Maharashtra.
- *Legal basis for the motorway defaults.* Motorways 7, 9 and 81 are special highways under the Highways Act B.E. 2535 (1992), s.7, which defines them by entry and exit only through supplemental roads. s.54 lets the special highway director forbid certain vehicle types or pedestrians on a special highway by gazette notification. The motorcycle and bicycle bans are attested by secondary sources citing the Land Traffic Act B.E. 2522 (1979), s.139 and the M81 prohibition list; the notification texts for M7 and M9 were not read (`src/road_access_rules.json`, `source.country_law.thailand`).
- *Statutory powers.* The power in s.54 sits in the Act's chapter on special highways and has no counterpart for national highways (s.8). So `highway=trunk` and below are open to every mode under the Highways Act, which matches the defaults. The Land Traffic Act was not read in full.

**Rows reached** (current build; the 75,830 rows come from 11,544 ADB segments, split by the TomTom layer and by the influence-zone and POI-zone steps):

| Row | V_safe | Rows | km |
|---|---|---|---|
| R0 | 30 | 74,865 | 59,210.8 |
| R1 | 50 | 0 | 0 |
| R2 | 30 | 64 | 47.3 |
| R3 | 70 | 5 | 10.6 |
| R4 | 80 | 160 | 134.7 |
| R4 | 90 | 165 | 186.4 |
| R4 | 100 | 559 | 986.8 |
| R5 | 50 | 12 | 1.4 |
| Total | | 75,830 | 60,577.9 |

Rows with `A` true, by `access_control_basis`: `osm_motorway` 454 (865.9 km; 449 in R4, 5 in R2), `osm_vru_prohibited` 325 (300.4 km; 281 in R4, 35 in R2, 6 in R5, 3 in R3), `overture_motorway_fallback` 135 (158.5 km; 110 in R4, 24 in R2, 1 in R3), `osm_motorroad` 51 (42.2 km; 44 in R4, 6 in R5, 1 in R3).

Lengths in both tables are the WGS84 geodesic length of each row's geometry.

## 7. What the Safe System sources require that the data cannot yet show

| Source requirement | Variable | Status |
|---|---|---|
| Physical barrier keeping pedestrians and cyclists off the carriageway (enables 50 km/h without legal exclusion) | `B_v` | Set from the Mapillary roadside objects; no row has it in the current build. |
| Physical barrier between motorcyclists and four-wheeled traffic (lets an access-controlled road that riders may use rise above 30 km/h) | `S` | Set from the Mapillary roadside objects; no row has it in the current build. 0 of the 71,629 OSM ways matched to ADB segments carry any `barrier=*` tag. |
| Full grade separation of every junction | `G` | Approximated by `bridge` / `tunnel` / `layer` on at least one matched way, which over-matches. |

## 8. How this table was checked

The formulas in §3 and §4, applied to the columns of `outputs/segments_priority.parquet`, reproduce `collision_type` on all 102,508 rows of the current build with no mismatch, give every R4 row the V_safe of its `road_class`, and put no row in more than one of R1 to R5. The same check confirms three properties the table relies on: `V` is false on every row where `A` is true; `A` is true exactly where `access_control_basis` is filled; and `J` is false on every row where `K` or `G` is true. Every row in R5 carries `v_safe_basis = side_impact:junction_buffer`.
