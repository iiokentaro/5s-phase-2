"""The production WaySource: osmium CLI for the read, duckdb for the parse.

Chosen over pyrosm because osmium was the ground truth pyrosm's two-node-only
output was measured against, and because it is faster by two orders
of magnitude: 2,915,689 Thailand highway ways exported in 17.6 s, where pyrosm
spent 3,775 s returning an incomplete 819,785.

The route is deliberately boring -- two osmium subprocesses and one SQL read --
because osm_way_source.assert_conservation is what establishes that it is
right, and a simple route is one whose failures the conservation check can
actually describe.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
import shapely

from osm_way_source import OSMIUM_BIN, OSMIUM_TIMEOUT, PROMOTED_TAGS

# osmium writes an RFC 8142 record separator (0x1e) before every line of
# `geojsonseq`. duckdb's read_ndjson returns every column NULL when it is
# present and reports no error, so it has to come off before the read.
RECORD_SEPARATOR = b"\x1e"


class OsmiumWaySource:
    name = "osmium"

    def extract(self, pbf_path: str | Path) -> gpd.GeoDataFrame:
        pbf_path = Path(pbf_path)
        with tempfile.TemporaryDirectory(prefix="osm_ways_") as tmp:
            seq = Path(tmp) / "ways.geojsonseq"
            ndjson = Path(tmp) / "ways.ndjson"
            self._export(pbf_path, seq)
            _strip_record_separator(seq, ndjson)
            return self._read(ndjson)

    # -- steps ------------------------------------------------------------- #

    @staticmethod
    def _export(pbf_path: Path, out: Path) -> None:
        """`-a id` puts the way id in properties as `@id`; without it the
        export carries tags only and the rows cannot be joined to anything.
        `--geometry-types=linestring` drops closed ways tagged area=yes, which
        is why assert_conservation allows the extract to sit slightly under the
        file's declared way count."""
        subprocess.run(
            [OSMIUM_BIN, "export", str(pbf_path), "-f", "geojsonseq",
             "--geometry-types=linestring", "-a", "id", "--overwrite", "-o", str(out)],
            check=True, capture_output=True, timeout=OSMIUM_TIMEOUT,
        )

    @staticmethod
    def _read(ndjson: Path) -> gpd.GeoDataFrame:
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")
        # Quoted identifiers throughout: `lanes:divided` is a legal OSM key and
        # an illegal bare SQL identifier, and renaming it here would put a name
        # into the frame that osm_way_source.WAY_COLUMNS does not know.
        selects = ",\n       ".join(
            "properties->>'{t}' AS \"{t}\"".format(t=tag) for tag in PROMOTED_TAGS
        )
        # ST_AsWKB keeps geometry construction inside duckdb; shapely.from_wkb
        # is vectorized, so the whole country materializes without a Python
        # loop over coordinates.
        table = con.execute(f"""
            SELECT CAST(properties->>'@id' AS BIGINT) AS osm_way_id,
                   {selects},
                   CAST(properties AS VARCHAR) AS tags,
                   ST_AsWKB(ST_GeomFromGeoJSON(geometry)) AS wkb
            FROM read_ndjson(
                '{ndjson}',
                columns={{'type':'VARCHAR','geometry':'JSON','properties':'JSON'}}
            )
        """).fetch_arrow_table()

        wkb = table.column("wkb").to_pylist()
        frame = pd.DataFrame({
            "osm_way_id": table.column("osm_way_id").to_pandas(),
            **{tag: table.column(tag).to_pandas() for tag in PROMOTED_TAGS},
            "tags": table.column("tags").to_pandas(),
        })
        return gpd.GeoDataFrame(frame, geometry=shapely.from_wkb(wkb), crs="EPSG:4326")


def _strip_record_separator(src: Path, dst: Path, chunk: int = 1 << 22) -> None:
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        while True:
            block = fin.read(chunk)
            if not block:
                break
            fout.write(block.replace(RECORD_SEPARATOR, b""))
    if os.path.getsize(dst) == 0:
        raise RuntimeError(f"{src} produced no rows -- osmium export wrote nothing")
