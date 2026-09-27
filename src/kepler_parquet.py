"""Write a GeoDataFrame as a GeoParquet kepler.gl can open.

kepler.gl's browser reader refuses large_string, dictionary and list columns
("arrow type not supported"). pandas 3 writes every `str` column as
large_string and an ordered Categorical as a dictionary, so both are turned
into plain strings before writing, and the written schema is checked for
anything that slipped through.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Arrow types kepler.gl's browser reader rejects.
_UNSUPPORTED_ARROW_TYPES = {
    "large_string": pa.types.is_large_string,
    "dictionary": pa.types.is_dictionary,
    "list": lambda t: pa.types.is_list(t) or pa.types.is_large_list(t),
}


def _is_text_column(s: pd.Series) -> bool:
    if isinstance(s.dtype, pd.CategoricalDtype):
        return pd.api.types.is_string_dtype(s.cat.categories)
    return pd.api.types.is_string_dtype(s)


def _is_list_column(s: pd.Series) -> bool:
    if s.dtype != object:
        return False
    return s.dropna().map(lambda v: isinstance(v, (list, tuple, np.ndarray))).any()


def write_kepler_parquet(
    gdf: gpd.GeoDataFrame, path, drop_lists: bool = False, compression: str = "snappy"
) -> str:
    """Write `gdf` to `path` with every text column as a plain `string`.

    With drop_lists, columns holding lists are dropped. Without it, a list
    column makes the write fail, so a caller that drops its known lists by name
    learns about a new one. compression is passed to GeoDataFrame.to_parquet.
    """
    export = gdf.copy()
    if drop_lists:
        export = export.drop(columns=[
            c for c in export.columns
            if c != export.geometry.name and _is_list_column(export[c])
        ])
    # An object column of Python strings is what pyarrow writes as `string`
    # (Utf8), the one text type kepler.gl reads.
    for col in export.columns:
        if col == export.geometry.name or not _is_text_column(export[col]):
            continue
        export[col] = export[col].astype(object)
        export[col] = export[col].where(export[col].notna(), None)

    export.to_parquet(path, compression=compression)

    schema = pq.read_schema(path)
    bad = [
        f"{field.name} ({name})"
        for field in schema if field.name != export.geometry.name
        for name, test in _UNSUPPORTED_ARROW_TYPES.items() if test(field.type)
    ]
    if bad:
        raise ValueError(f"{path} has columns kepler.gl cannot read: {', '.join(bad)}")
    return str(path)
