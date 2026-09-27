# Valhalla Pedestrian Isochrone Environment

## For reviewers (choosing a reproduction path)

| Path | Valhalla required? | Time |
|---|---|---|
| **Fast path (recommended)** `quick_reproduce.py` | **No** | ~10 seconds |
| Only if you want to **regenerate** isochrones | **Yes** | First tile build 30 min–2 hours + isochrone generation 15–45 min |

The school isochrones at the default minutes (`data/processed/isochrones/school_*_u3_r5.parquet`) are committed. `quick_reproduce.py` and `build_v_safe.py` read these files directly, so with the default POI parameters **everything works fully without starting Valhalla**.

From the map sidebar (`python src/serve_map.py --enable-pipeline`), the Full and Complete presets start and stop these containers themselves (`src/valhalla_service.py`) whenever the POI parameters need an isochrone that is not cached: another POI type, or other walking minutes. Docker Desktop has to be running.

The steps below are **for running Valhalla by hand**.

---

## Prerequisites

- **Docker Desktop** (Mac/Windows) or Docker Engine + Compose plugin (Linux)
 - Memory: 8 GB or more recommended for Thailand PBF processing (adjust under Docker Desktop → Settings → Resources)
 - Disk: 5–10 GB free recommended for tile cache
- `data/external/thailand-260621.osm.pbf` and `data/external/western-zone-260621.osm.pbf` must be present
 - **Use the full pbf here.** `data/external/*.5s-filtered.osm.pbf`, produced by `src/prefilter_pbf.py` for the pyrosm extraction, is a tag-filtered subset with no complete routable network and must not be substituted -- tiles built from it would silently produce wrong isochrones.
 - These are `.gitignore`d and not committed. Download URLs are documented in comments in `src/exposure_signals.py`

---

## Setup

### 1. Raise Docker Desktop memory limit (Mac/Windows)

Docker Desktop → Settings → Resources → Memory: set to **8 GB or more**.
The Thailand PBF consumes a large amount of memory during tile building.

### 2. Build tiles for both countries (run in two separate steps)

Run from the `valhalla/` directory.

```bash
cd valhalla

# --- Thailand ---
docker compose --profile thailand up
# Wait until you see "valhalla_service is Running" (first run: 30 min–1.5 hours)
# Tiles are saved under valhalla/tiles/thailand/ (.gitignore)

# In another terminal, generate isochrones
cd..
python src/poi_isochrone.py --country thailand --poi-type school
# When done, stop Valhalla
cd valhalla && docker compose --profile thailand down

# --- Maharashtra ---
docker compose --profile maharashtra up
# Wait until you see "valhalla_service is Running" (first run: 10–30 min)

# In another terminal, generate isochrones
cd..
python src/poi_isochrone.py --country maharashtra --poi-type school
# When done, stop Valhalla
cd valhalla && docker compose --profile maharashtra down
```

### 3. Verify isochrone parquet files

```bash
python - <<'EOF'
import geopandas as gpd
for c in ["thailand", "maharashtra"]:
 gdf = gpd.read_parquet(f"data/processed/isochrones/school_{c}_u3_r5.parquet")
 print(f"{c}: {len(gdf)} rows, source breakdown:", gdf["source"].value_counts.to_dict)
EOF
```

### 4. Re-run the full pipeline

```bash
python src/build_v_safe.py
python src/quick_reproduce.py
```

---

## Subsequent runs (after tiles are built)

If tiles exist under `valhalla/tiles/{country}/`, the server starts immediately after the container launches (tile rebuild is skipped).

```bash
cd valhalla
docker compose --profile thailand up # starts in seconds to ~1 minute
```

If isochrone parquet already exists, `poi_isochrone.py` returns the cache and exits immediately (idempotent). To regenerate, pass `--force`:

```bash
python src/poi_isochrone.py --country thailand --poi-type school --force
```

`--poi-type` is one of school, hospital, marketplace, shop, bus_stop, and `--minutes-urban` / `--minutes-rural` set the walking time (default 3 / 5). Each type, country and pair of minutes is its own file, `data/processed/isochrones/{type}_{country}_u{urban}_r{rural}.parquet`.

The origins are every candidate POI of the type (`src/poi_sources.py:poi_candidates`: all OSM ones plus every fetched Overture one, before the confidence rule): for schools, 60,384 for Thailand and 17,216 for Maharashtra. Each build uses the subset its POI parameters select, so changing the confidence values never needs Valhalla. Regenerate the isochrones when `osm_pois_{country}.parquet` or `overture_pois_{country}.parquet` changes; the pipeline's `poi_isochrones` step does so on its own when an isochrone parquet is older than either. With the tiles already built, generation takes about 5 minutes for Thailand and about 1.5 minutes for Maharashtra (16 parallel requests).

---

## Health check

Once Valhalla is running, you can verify it with:

```bash
# Thailand: 3-minute pedestrian isochrone from central Bangkok
curl -s -X POST http://localhost:8003/isochrone \
 -H 'Content-Type: application/json' \
 -d '{"locations":[{"lat":13.7563,"lon":100.5018}],"costing":"pedestrian","contours":[{"time":3}],"polygons":true}' \
 | python -m json.tool | head -20
```

If the response is JSON containing `"type": "FeatureCollection"`, the service is healthy.

---

## Directory layout

```
valhalla/
├── docker-compose.yml # Two profiles: Thailand / Maharashtra
├── README.md # This file
├── tiles/
│ ├── thailand/ # Thailand tiles (.gitignore)
│ └── maharashtra/ # Maharashtra tiles (.gitignore)
└── custom_files/ # Manual copy target (docker-compose mounts PBF directly)
 └──.gitkeep
```

`data/external/*.osm.pbf` is mounted into the container via `volumes:` in docker-compose. Manual copying into `valhalla/custom_files/` is not required.

---

## Parameters (`src/poi_isochrone.py`)

The walking minutes are POI parameters (`src/poi_params.py`, `iso_min_urban` / `iso_min_rural` per type, default 3 / 5, 1 to 30).

| Constant | Default | Meaning |
|---|---|---|
| `RURAL_FLOOR_M` | 200 | Rural safety-floor circular buffer radius (m) |
| `ISOCHRONE_CORRIDOR_M` | 15 | Degeneracy-fix corridor width (m) |
| `MAX_SNAP_M` | 50 | Maximum network snap distance (m) |
| `PARALLEL_WORKERS` | 16 | Parallel Valhalla API requests |
| `VALHALLA_URL` | `http://localhost:8003/isochrone` | Valhalla endpoint |
| `MAX_FALLBACK_SHARE` | 0.5 | Above this share of `buffer_fallback`, the build stops without writing (Valhalla is not serving that country) |

---

## Troubleshooting

### Tile build stops mid-way / OOM

Increase Docker Desktop memory limit (8 GB recommended).

### `docker compose` command not found

Install the latest Docker Desktop, or try `docker-compose` (with a hyphen).

### Cannot connect to `localhost:8003`

```bash
docker compose --profile thailand ps # confirm Running
docker compose --profile thailand logs # check error logs
```

### All isochrones become `buffer_fallback`

Confirm Valhalla responds at `localhost:8003` with `curl`. If many `snapped=False` results appear, increase `MAX_SNAP_M` (or use the `--max-snap-m` option), or check whether OSM road coverage is sparse in that area.

### Missing `data/external/*.osm.pbf`

OSM PBF files are `.gitignore`d and not committed. Create `data/external/` and place the PBF files there. Sources are listed in `PBF_PATHS` at the top of `exposure_signals.py` (fetch the relevant regional PBF from Geofabrik, etc.). That constant still points at the full pbfs; do not point Valhalla at the `*.5s-filtered.osm.pbf` derivatives.

---

## Design notes

- **Tile persistence:** Tiles are cached under `valhalla/tiles/{country}/`, so tile rebuild is skipped after container restarts.
- **Separate builds per country:** Thailand and Maharashtra PBFs can be processed in one container, but memory use grows. Running profiles separately is more stable.
- **Committing isochrone parquet:** Committing `data/processed/isochrones/school_*_u3_r5.parquet` lets reviewers run `quick_reproduce.py` without Valhalla. If file size exceeds 100 MB, consider Git LFS or documenting regeneration steps in the README only.
